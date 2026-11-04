"""SCAGE exploration in an Open3D simulator.

    python scage_explore.py --mesh examples/mv2HUxq3B53.glb --auto_start \
        --max_steps 1500 --dump out/run1 --exit_when_done

Two windows, as in FrontierNet's demo: an observer view, and the robot's
camera. The robot's pose *is* that window's camera, so a mouse drag or a key
in the robot window moves the robot - keep out of it during a run. Every
frame's depth (to 8 m) goes into two maps, every step: Wavemap for navigation
and the 0.05 m Delta-Net point map. Every `predict_interval`
steps, or on arrival, Delta-Net scores the whole point map, anomalous points
are clustered and each cluster gets a safe camera target; nearby targets are
clubbed into one view and a short tour is picked by set cover on
g * S / max(d, eps) (paper Eq. 4 with travel distance; scage/selection.py).

The robot commits to its goal: at each later Delta-Net run the goal is kept
unless what it was going to look at has resolved (it sees under keep_frac of
its anomaly mass) or it is no longer safe. On arrival the rest of the tour
comes first. The robot is a UAV confined to the building (scage/space.py);
every step is re-checked against the newest map.

Run with --dump DIR to keep a .ply snapshot and a JSON line per update, the
pose of every frame (frames.npy) and a run summary, for tools/eval_run.py.
With --write_path RUN.json the end of the run also leaves RUN_cloud.ply, the
Delta-Net map in colour, and RUN_path.ply, the robot's path (scage.dump).
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch

import scage  # noqa: F401  (path setup)
from infer import ramp
from scage import load_config
from scage.dump import write_final, write_update
from scage.pipeline import ScagePipeline
from scage.viewer import (camera_vis_with_cylinders, capture_depth, capture_rgb, create_camera,
                          create_interactive_vis, get_vis_state, is_vis_moving, load_mesh,
                          register_basic_callbacks, set_vis_cam_ex, set_vis_cam_intr)
from scage.viewpoints import camera_pose, look_around


@dataclass
class Stop:
    """A view the robot is going to, or will go to next."""
    pose: np.ndarray
    mass0: float            # anomaly mass (m^2) it saw when chosen
    g: float
    kind: str
    rejects: int = 0


class SCAGEApp:
    REFRESH_RATE = 50                           # Hz, the window loop after a run
    CAM1_H, CAM1_W, CAM1_F = 960, 1280, 700.0   # observer window
    CAM2_H, CAM2_W, CAM2_F = 480, 480, 300.0    # robot camera, 77.32 deg FOV as in the paper

    MAX_PLAN_TRIES = 5          # next-best views to try before giving up on a replan
    LOOK_AROUND_STEPS = 8       # in-place yaw steps when nothing is reachable
    LOOK_AROUND_PITCH = 30.0    # degrees up: a level camera never sees the ceiling
                                # within ~2 m of itself, and the roof map needs it
    SHOW_TARGETS = 15           # targets drawn in the observer window
    OVERLAY_POINT_SIZE = 2.0    # px, the scored cloud in the observer window
    OVERLAY_KEEP = 0.3          # share of its low-score points drawn, so the robot
                                # and the scene show through (anomalous ones all are)

    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        self.predict_interval = int(self.config.get("predict_interval", 5))
        self.vis_1 = None                   # observer window
        self.vis_2 = None                   # robot camera
        self.mapper = None
        self.ft_manager = None
        self.geometry_vis_1 = []            # the robot's frustum in the observer window
        self.ft_geometry_vis_1 = []         # the targets' frustums there
        self.path_to_go = []
        self.move_enough = True
        self.last_W_T_C2 = np.eye(4)        # robot pose at the last recorded step
        self.json_path = args.write_path or None
        self._lock = threading.Lock()
        self.pipeline = None
        self.updates = []
        self.overlay = None                 # scored cloud in the observer window
        self.step_clouds = []               # points added since, one cloud per step
        self.shown = 0
        self.last_detect_step = None
        self.looked_around = False
        self.goal = None
        self.tour_rest = []
        self.decision = {}
        self.counts = Counter()
        self.idle = 0
        self.last_n = None
        self.unsafe_steps = 0
        sc = self.config["scage"]
        self.keep_frac = float(sc["keep_frac"])
        self.min_mass = float(sc["min_mass"])
        self.max_rejects = int(sc["max_rejects"])
        self.idle_limit = int(sc["idle_limit"])
        self.tour_len = int(sc["tour_len"])
        self.fov_deg = 2 * np.degrees(np.arctan(self.CAM2_W / (2 * self.CAM2_F)))
        if args.start:
            x, y, z, yaw = map(float, args.start.split(","))
            yaw = np.radians(yaw)
            W_T_C = camera_pose([x, y, z], [np.cos(yaw), np.sin(yaw), 0.0])
            self.config["initial_cam_extrinsic"] = np.linalg.inv(W_T_C).tolist()
        self.dump_dir = args.dump
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

    # ---- setup --------------------------------------------------------------

    def setup_viewers(self):
        """Create both windows, add the scene mesh, place the cameras."""
        # Logging level
        if self.args.log_level < 10:
            logging.getLogger().setLevel(logging.NOTSET)
        elif self.args.log_level < 20:
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.args.log_level < 30:
            logging.getLogger().setLevel(logging.INFO)
        elif self.args.log_level < 40:
            logging.getLogger().setLevel(logging.WARNING)
        elif self.args.log_level < 50:
            logging.getLogger().setLevel(logging.ERROR)
        else:
            logging.getLogger().setLevel(logging.CRITICAL)

        # Cameras
        cam_intr_1 = create_camera(self.CAM1_H, self.CAM1_W, self.CAM1_F)
        cam_intr_2 = create_camera(self.CAM2_H, self.CAM2_W, self.CAM2_F)

        self.vis_1 = create_interactive_vis(
            self.CAM1_H,
            self.CAM1_W,
            cam_intr_1,
            show_back_face=False,
            light_on=False,
            z_near=0.02,
            z_far=50.0,
        )
        self.vis_2 = create_interactive_vis(
            self.CAM2_H,
            self.CAM2_W,
            cam_intr_2,
            show_back_face=True,
            light_on=False,
            z_near=0.02,
            z_far=50.0,
        )

        # Load scene mesh
        scene_mesh = load_mesh(self.args.mesh)
        self.vis_1.add_geometry(scene_mesh, reset_bounding_box=True)
        self.vis_2.add_geometry(scene_mesh, reset_bounding_box=True)

        # Initial frustum of vis_2, drawn in vis_1
        C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]  # (W→C2)
        frustum = camera_vis_with_cylinders(
            C2_T_W,
            wh_ratio=self.CAM2_W / self.CAM2_H,
            scale=0.8,
            weight=0.0,
            fovx=2 * np.degrees(np.arctan(self.CAM2_W / (2 * self.CAM2_F))),
            radius=0.04,
        )
        for g in frustum:
            self.geometry_vis_1.append(g)
            self.vis_1.add_geometry(g)

        # Global axis
        world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        self.vis_1.add_geometry(world_axis, reset_bounding_box=False)

        # The overlay is a screen door: legacy Open3D draws points opaque
        self.vis_1.get_render_option().point_size = self.OVERLAY_POINT_SIZE

        # Register basic callbacks
        register_basic_callbacks(self.vis_1)
        register_basic_callbacks(self.vis_2)

        # Reset intrinsics
        set_vis_cam_intr(self.vis_1, cam_intr_1)
        set_vis_cam_intr(self.vis_2, cam_intr_2)

        # Observer & robot poses from config
        obs_C_T_W = np.asarray(self.config["observer_cam_extrinsic"], dtype=float)
        rob_C_T_W = np.asarray(self.config["initial_cam_extrinsic"], dtype=float)
        set_vis_cam_ex(self.vis_1, obs_C_T_W)
        set_vis_cam_ex(self.vis_2, rob_C_T_W)

        # Open3D can drop a camera set before the window's first poll; set both
        # again after one and check the robot really starts where configured
        for _ in range(3):
            for v in (self.vis_1, self.vis_2):
                v.poll_events()
                v.update_renderer()
        set_vis_cam_intr(self.vis_2, create_camera(self.CAM2_H, self.CAM2_W, self.CAM2_F))
        rob_C_T_W = np.asarray(self.config["initial_cam_extrinsic"], float)
        set_vis_cam_ex(self.vis_1, np.asarray(self.config["observer_cam_extrinsic"], float))
        set_vis_cam_ex(self.vis_2, rob_C_T_W)
        for v in (self.vis_1, self.vis_2):
            v.poll_events()
            v.update_renderer()
        assert np.allclose(get_vis_state(self.vis_2)["cam_extrinsic"], rob_C_T_W, atol=1e-5), \
            "robot camera did not take the configured start pose"

    def setup_system(self):
        intr = get_vis_state(self.vis_2)["cam_intrinsic"]
        self.pipeline = ScagePipeline(
            self.config, intr.intrinsic_matrix, intr.height, intr.width,
            device="cuda" if torch.cuda.is_available() else "cpu",
            log_level=self.args.log_level)
        self.mapper = self.pipeline.mapper          # DualMapper: move() feeds both maps
        self.ft_manager = self.pipeline.manager

    def _robot_pose(self):
        return np.linalg.inv(get_vis_state(self.vis_2)["cam_extrinsic"])

    def get_rgbd(self, vis, return_rgb_type="np", return_depth_type="np"):
        """RGB and depth from a window (depth is the renderer's, ground truth)."""
        rgb = capture_rgb(vis, return_rgb_type=return_rgb_type)
        img = rgb["image"]
        if np.max(img) <= 1:
            img = (img * 255).astype(np.uint8)
        depth = capture_depth(vis, return_depth_type=return_depth_type)["depth"]
        return img, depth

    # ---- main loop ------------------------------------------------------------

    def exploration(self):
        p, m = self.pipeline, self.ft_manager
        self.vis_1.update_renderer()
        self.vis_2.update_renderer()
        start = time.time()

        W_T_C = self._robot_pose()
        rgb, depth = self.get_rgbd(self.vis_2)
        p.observe(depth, W_T_C, rgb)
        m.add_robot_poses([W_T_C])
        self.last_W_T_C2 = W_T_C
        reason = "max steps"

        while True:
            W_T_C = self._robot_pose()
            n = len(m.robot_poses)
            logging.info(" -------Current exploration step: %d -------", n)
            if n > self.args.max_steps:
                break
            if time.time() - start > self.args.max_time:
                reason = "time limit"
                break
            # a loop that keeps replanning without the robot moving
            self.idle = 0 if n != self.last_n else self.idle + 1
            self.last_n = n
            if self.idle >= self.idle_limit and self.goal is not None:
                logging.warning("No progress in %d iterations; giving up on the goal.", self.idle)
                self._give_up("idle")
                self.idle = 0

            due = (not self.path_to_go or self.last_detect_step is None
                   or n - self.last_detect_step >= self.predict_interval)
            if due:
                rep = self._detect(n, W_T_C)
                stop = self._decide(W_T_C)
                rep["decision"] = self.decision
                self._log(rep)
                if self.dump_dir:
                    write_update(self.dump_dir, len(self.updates) - 1, p, W_T_C, self.fov_deg,
                                 None if self.goal is None else self.goal.pose)
                if stop:
                    reason = stop
                    break
            else:
                # both maps grew this step; the navigation map and flight space
                # follow, for the per-step safety check in move()
                p.update_nav_map()

            if self.json_path:
                m.write_to_file(self.json_path)
            if self.path_to_go:
                self.move(steps=1)

        logging.info("Exploration finished (%s), total steps: %d", reason, n)
        self._finish(reason, time.time() - start)

    def _detect(self, n, W_T_C):
        p, m = self.pipeline, self.ft_manager
        rep = p.detect()
        rep["select"] = p.select(W_T_C[:3, 3])
        self.last_detect_step = n
        rep.update(update=len(self.updates), step=n, valid=len(m.valid_frontiers),
                   robot=W_T_C[:3, 3].tolist(), planner=dict(m.planner.stats),
                   plans=dict(m.plan_stats))
        sel = rep["select"]
        logging.info(
            "Delta-Net update %d: %d pts, %d anomalous, %d clusters -> %d safe targets -> "
            "%d views (%s), %d unreachable; tour %s; score %.1fs select %.1fs",
            rep["update"], rep["points"], rep["anomalous"], rep["clusters"], rep["targets"],
            rep["valid"], dict(sel["kinds"]), sel["unreachable"],
            [(t["d"], t["mass"]) for t in sel["tour"]], rep["t_score"], sel["t_select"])
        self.updates.append(rep)
        self._show_anomalies()
        return rep

    def _log(self, rep):
        if self.dump_dir:
            with open(os.path.join(self.dump_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps(rep, default=float) + "\n")

    def _stop(self, ft):
        return Stop(pose=ft.pose6d.copy(), mass0=float(ft.features.get("mass", 0.0)),
                    g=float(ft.u_gain or 0.0), kind=str(ft.features.get("kind")))

    def _worth(self, st):
        """Is a stop still worth going to? -> (ok, why, anomaly mass it sees now)"""
        if self.ft_manager.blacklisted(st.pose):
            return False, "blacklisted", 0.0
        why = self.pipeline.viewpoints.check(st.pose[:3, 3])
        if why is not None:
            return False, why, 0.0
        mass = self.pipeline.view_mass(st.pose)
        if mass < max(self.keep_frac * st.mass0, self.min_mass):
            return False, "resolved", mass
        return True, None, mass

    def _decide(self, W_T_C):
        """Keep the committed goal, or pick the next stop and plan to it.
        -> a reason to stop exploring, or None."""
        m = self.ft_manager
        if self.goal is not None and self.path_to_go:
            ok, why, mass = self._worth(self.goal)
            if ok:
                self.decision = dict(action="keep", mass=round(mass, 4),
                                     mass0=round(self.goal.mass0, 4))
                return None
            logging.info("Dropping the goal: %s (sees %.3f of %.3f m^2).", why, mass,
                         self.goal.mass0)
            self.counts[f"goal_dropped_{why}"] += 1
            self.goal, self.path_to_go = None, []

        # the rest of the committed tour first, then this run's tour, then the rest
        tour = [st for st in self.tour_rest if self._worth(st)[0]]
        tour += [self._stop(m.frontiers[f]) for f in m.tour if f in m.frontiers]
        rest = sorted((ft for ft in m.valid_frontiers if ft.id not in m.tour),
                      key=lambda f: -(f.utility or 0.0))
        queue, seen = [], []
        for st in tour + [self._stop(ft) for ft in rest]:
            if m.blacklisted(st.pose) or any(
                    np.linalg.norm(st.pose[:3, 3] - q[:3, 3]) < m.group_trans and
                    st.pose[:3, 2] @ q[:3, 2] > np.cos(m.group_angle) for q in seen):
                continue
            queue.append(st)
            seen.append(st.pose)
        n_tour = sum(1 for st in queue if any(st is t for t in tour))

        for k, st in enumerate(queue[:self.MAX_PLAN_TRIES]):
            path, how = m.plan_to(st.pose, W_T_C)
            if path:
                self.goal, self.path_to_go = st, path
                self.move_enough, self.looked_around = False, False
                self.tour_rest = queue[k + 1:n_tour][:self.tour_len - 1] if k < n_tour else []
                self.decision = dict(action="plan", how=how, tried=k + 1, kind=st.kind,
                                     mass0=round(st.mass0, 4), steps=len(path),
                                     goal=st.pose[:3, 3].round(3).tolist(),
                                     tour_rest=len(self.tour_rest))
                logging.info("Path to goal found with %d steps (%s).", len(path), how)
                return None
            logging.warning("No path to a view; blacklisting it.")
            m.blacklist(st.pose)
            self.counts["blacklisted_unplannable"] += 1

        self.goal, self.tour_rest = None, []
        if self.looked_around:
            self.decision = dict(action="stop", views=len(queue))
            return "no reachable view left" if queue else "no anomaly left worth visiting"
        logging.info("No reachable view; looking around in place.")
        self.path_to_go = self._look_around(W_T_C)
        self.looked_around = True
        self.counts["look_arounds"] += 1
        self.decision = dict(action="look_around", views=len(queue))
        return None

    def _give_up(self, why):
        if self.goal is not None:
            self.ft_manager.blacklist(self.goal.pose)
            self.counts[f"blacklisted_{why}"] += 1
        self.goal, self.path_to_go = None, []
        self.move_enough = True

    def _look_around(self, W_T_C):
        return look_around(W_T_C, self.LOOK_AROUND_STEPS, self.LOOK_AROUND_PITCH)

    def move(self, steps):
        """Follow the path, after checking the straight step to the next pose
        against the newest map - the path was planned against an older one."""
        if self.path_to_go and self.goal is not None:
            m, cur = self.ft_manager, self._robot_pose()
            nxt = self.path_to_go[0][:3, 3]
            sp = m.space
            skip = 0.0 if sp.valid(cur[:3, 3]) else 0.25
            if np.linalg.norm(nxt - cur[:3, 3]) > 0.05 and not sp.segment_valid(
                    cur[:3, 3], nxt, skip_start=skip):
                self.unsafe_steps += 1
                self.goal.rejects += 1
                L = np.linalg.norm(nxt - cur[:3, 3])
                ts = np.linspace(0, 1, max(int(np.ceil(L / 0.05)), 1) + 1)
                pts = cur[:3, 3] + np.outer(ts, nxt - cur[:3, 3])
                bad = [(round(t * L, 2), sp.reason(q)) for t, q in zip(ts, pts)
                       if t * L >= skip and not sp.valid(q)][:3]
                logging.warning("Next step %s -> %s is no longer safe in the updated map "
                                "(%s; %d/%d); replanning to the same goal.",
                                cur[:3, 3].round(2), nxt.round(2), bad, self.goal.rejects,
                                self.max_rejects)
                path = None
                if self.goal.rejects < self.max_rejects:
                    path, _ = m.plan_to(self.goal.pose, cur)
                if path:
                    self.path_to_go = path
                else:
                    self._give_up("unsafe")
                return
        self._follow(steps)
        self._show_new_points()

    def _follow(self, steps):
        """Up to `steps` poses along the path: set the robot camera, capture,
        insert into the maps, and record the pose once it has moved enough."""
        if self.vis_1 is None or self.vis_2 is None:
            return
        if not self.path_to_go:
            logging.info("No path to follow.")
            return

        C1_T_W = get_vis_state(self.vis_1)["cam_extrinsic"]

        for _ in range(steps):
            if not self.path_to_go:
                logging.info("Path exhausted.")
                break

            next_W_T_C = self.path_to_go.pop(0)
            logging.debug(f"Moving to next pose:\n{next_W_T_C}")

            # Update vis_2 camera extrinsic
            set_vis_cam_ex(self.vis_2, np.linalg.inv(next_W_T_C))

            # Capture new depth (and colour, for the saved cloud)
            rgb, depth = self.get_rgbd(self.vis_2)

            # Insert into mapper
            C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]
            W_T_C2 = np.linalg.inv(C2_T_W)
            self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C2, rgb=rgb)

            # Update viewers
            self.vis_1.update_renderer()
            self.vis_2.update_renderer()
            self.vis_1.poll_events()
            self.vis_2.poll_events()
            time.sleep(0.1)

            # If we truly moved, update path bookkeeping
            if is_vis_moving(
                self.vis_2, self.last_W_T_C2, trans_thre=0.1, rot_thre=0.26
            ):
                self.last_W_T_C2 = W_T_C2
                if self.ft_manager is not None:
                    self.ft_manager.add_robot_poses([W_T_C2])
                self.move_enough = True

        # Refresh overlays
        self.update_geometry_vis_1()
        self.vis_1.update_renderer()
        self.vis_2.update_renderer()
        set_vis_cam_ex(self.vis_1, C1_T_W)

    # ---- output -------------------------------------------------------------

    def update_vis(self):
        """Redraw both windows, keeping their cameras."""
        if self.vis_1 is None or self.vis_2 is None:
            return

        cam_ex_1 = get_vis_state(self.vis_1)["cam_extrinsic"]
        cam_ex_2 = get_vis_state(self.vis_2)["cam_extrinsic"]

        # If vis_2 moved enough, refresh frustum and overlays
        if is_vis_moving(self.vis_2, self.last_W_T_C2, trans_thre=0.1, rot_thre=0.26):
            # Store latest
            self.last_W_T_C2 = np.linalg.inv(cam_ex_2)
            self.update_geometry_vis_1()

        # Render and restore cams
        self.vis_1.update_renderer()
        self.vis_2.update_renderer()
        set_vis_cam_ex(self.vis_1, cam_ex_1)
        set_vis_cam_ex(self.vis_2, cam_ex_2)

    def update_geometry_vis_1(self):
        """The robot's frustum (blue) and the targets, in the observer window."""
        if self.vis_1 is None or self.vis_2 is None:
            return

        # vis_2 camera poses
        C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]
        W_T_C2 = np.linalg.inv(C2_T_W)

        intr = get_vis_state(self.vis_2)["cam_intrinsic"]
        W2, H2 = intr.width, intr.height
        fx2 = intr.intrinsic_matrix[0, 0]

        wh_ratio = W2 / H2
        fovx_deg = 2.0 * np.degrees(np.arctan(W2 / (2.0 * fx2)))

        frustum_meshes = camera_vis_with_cylinders(
            W_T_C2,
            wh_ratio=wh_ratio,
            scale=0.8,
            weight=0.0,
            color=[0, 0, 1],
            fovx=fovx_deg,
            radius=0.04,
        )

        # Clear previous
        for g in self.geometry_vis_1:
            self.vis_1.remove_geometry(g)
        self.geometry_vis_1.clear()

        # Frontiers overlay
        self.visualize_all_frontiers(self.vis_1)

        # Frustum overlay
        for g in frustum_meshes:
            self.geometry_vis_1.append(g)
            self.vis_1.add_geometry(g, reset_bounding_box=False)

    def visualize_all_frontiers(self, vis):
        """The best SHOW_TARGETS views (yellow -> red with utility; drawing all
        ~100 of them makes every Open3D redraw take seconds), the tour in green
        and the committed goal in blue."""
        m = self.ft_manager
        C_T_W = get_vis_state(vis)["cam_extrinsic"]
        for g in self.ft_geometry_vis_1:
            vis.remove_geometry(g, reset_bounding_box=False)
        self.ft_geometry_vis_1.clear()
        fts = sorted(m.valid_frontiers, key=lambda f: -(f.utility or 0.0))[:self.SHOW_TARGETS]
        umax = max([f.utility or 0.0 for f in fts] + [1e-9])
        draw = [(ft.pose6d, (ft.utility or 0.0) / umax,
                 [0.1, 0.8, 0.1] if ft.id in m.tour else None) for ft in fts]
        if self.goal is not None:
            draw.append((self.goal.pose, 0.0, [0.0, 0.0, 1.0]))
        for pose, w, color in draw:
            for g in camera_vis_with_cylinders(pose, wh_ratio=1.0, scale=0.6,
                                               fovx=self.fov_deg, radius=0.04,
                                               weight=w, color=color):
                self.ft_geometry_vis_1.append(g)
                vis.add_geometry(g, reset_bounding_box=False)
        set_vis_cam_ex(vis, C_T_W)

    def _screen_door(self, rows):
        """The OVERLAY_KEEP of the map's rows drawn: evenly spread (golden-ratio
        sequence) and fixed, since rows never move - the map is append-only."""
        return (rows * 0.6180339887498949) % 1.0 < self.OVERLAY_KEEP

    def _show_anomalies(self):
        """The whole map, coloured by the new scores (replaces the per-step clouds):
        every anomalous point, and a screen door of the rest."""
        pm, s = self.pipeline.points, self.pipeline.last["score"]
        show = (s > self.pipeline.sc["tau"]) | self._screen_door(np.arange(len(pm)))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pm.xyz[show].astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(ramp(s[show]))
        for g in self.step_clouds + ([self.overlay] if self.overlay is not None else []):
            self.vis_1.remove_geometry(g, reset_bounding_box=False)
        self.step_clouds = []
        self.vis_1.add_geometry(pc, reset_bounding_box=False)
        self.overlay = pc
        self.shown = len(pm)

    def _show_new_points(self):
        """Points the last step added to the Delta-Net map, grey until scored
        (a screen door of them)."""
        pm = self.pipeline.points
        rows = np.arange(self.shown, len(pm))
        rows = rows[self._screen_door(rows)]
        self.shown = len(pm)
        if not len(rows):
            return
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pm.xyz[rows].astype(np.float64))
        pc.paint_uniform_color([0.6, 0.6, 0.6])
        C_T_W = get_vis_state(self.vis_1)["cam_extrinsic"]
        self.vis_1.add_geometry(pc, reset_bounding_box=False)
        set_vis_cam_ex(self.vis_1, C_T_W)
        self.step_clouds.append(pc)

    def _finish(self, reason, seconds):
        if self.json_path:
            files = write_final(os.path.splitext(self.json_path)[0], self.pipeline.points,
                                self.mapper.frames)
            logging.info("Saved %s", ", ".join(files))
        if not self.dump_dir:
            return
        m = self.ft_manager
        frames = np.stack(self.mapper.frames)
        robot = np.stack([m.robot_poses[k] for k in sorted(m.robot_poses)])
        np.save(os.path.join(self.dump_dir, "frames.npy"), frames)
        np.save(os.path.join(self.dump_dir, "robot_poses.npy"), robot)
        summary = dict(reason=reason, seconds=seconds, steps=len(robot), frames=len(frames),
                       updates=len(self.updates), points=len(self.pipeline.points),
                       unsafe_steps=self.unsafe_steps, blacklisted=len(m.unreachable),
                       counts=dict(self.counts), planner=dict(m.planner.stats),
                       plans=dict(m.plan_stats),
                       outside_own_map=int((~m.space.indoors(robot[:, :3, 3])).sum()),
                       path_length=float(np.linalg.norm(np.diff(frames[:, :3, 3], axis=0),
                                                        axis=1).sum()))
        with open(os.path.join(self.dump_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        logging.info("Run summary: %s", summary)

    def run(self):
        if self.args.auto_start and self.args.exit_when_done:
            self.setup_viewers()
            self.setup_system()
            try:
                self.exploration()
            finally:
                self.vis_1.destroy_window()
                self.vis_2.destroy_window()
            return

        self.setup_viewers()
        self.setup_system()
        if self.args.auto_start:
            self.exploration()
        else:
            logging.info(" --- Manual mode: move the robot camera with WASD (translation), "
                         "JL (rotation), QZ (height) in the small window --- ")

            def on_space(vis):
                self.exploration()
                return False

            self.vis_1.register_key_callback(32, on_space)
            self.vis_2.register_key_callback(32, on_space)
            logging.info(" --- PRESS SPACE TO START EXPLORATION --- ")

        try:
            while self.vis_1.poll_events() and self.vis_2.poll_events():
                with self._lock:
                    self.update_vis()
                time.sleep(1.0 / self.REFRESH_RATE)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.vis_1.destroy_window()
            self.vis_2.destroy_window()


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=str, required=True, help="Path to the mesh file")
    p.add_argument("--config", type=str, default=scage.CONFIG, help="SCAGE configuration file")
    p.add_argument("--write_path", type=str,
                   help="JSON-lines file: targets and robot poses, appended every step; "
                        "at the end also <name>_cloud.ply (the coloured map) and "
                        "<name>_path.ply (the robot's path) beside it")
    p.add_argument("--auto_start", action="store_true", default=False,
                   help="start exploring at once instead of waiting for SPACE")
    p.add_argument("--max_steps", type=int, default=150,
                   help="Maximum number of exploration steps")
    p.add_argument("--max_time", type=int, default=3600,
                   help="Maximum exploration time in seconds")
    p.add_argument("--log_level", "-ll", type=int, default=20,
                   help="logging level (0=notset, 10=debug, 20=info...)")
    p.add_argument("--dump", type=str, default=None,
                   help="directory for per-update .ply snapshots, logs and frame poses")
    p.add_argument("--start", type=str, default=None,
                   help="start pose 'x,y,z,yaw_deg' instead of the config's")
    p.add_argument("--exit_when_done", action="store_true",
                   help="with --auto_start, close the windows and exit when finished")
    return p


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
                        datefmt="%H:%M:%S", level=logging.WARNING)
    args = build_arg_parser().parse_args()
    code = 0
    try:
        SCAGEApp(args).run()
    except Exception:
        logging.exception("SCAGE run failed")
        code = 1
    # the ompl 1.7.0 wheel crashes in its static destructors at interpreter exit
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
