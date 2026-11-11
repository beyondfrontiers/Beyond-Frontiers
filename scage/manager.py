"""Target bookkeeping, gain, visibility and planning (paper Sec. III-B, Eq. 4).

Targets are `Target`s - FrontierNet's Frontier, trimmed: pos3d is the camera
position, view_direction its optical axis, pose6d the full camera pose. Their
bookkeeping (ids, the gain / bbox / pitch / clearance filter, robot poses) is
FrontierNet's FrontierManager's, without its graph. What differs from
FrontierNet:

- the target set is replaced wholesale after every Delta-Net run, since each run
  scores the whole map;
- the gain g(T) is the unknown volume the target camera would see, raycast
  against the current maps, rather than a network's guess reduced by what is
  already known;
- which anomalous points a view sees is raycast the same way; scage.selection
  uses it to club nearby views and pick a tour (Eq. 4 with travel distance);
- where the robot may be is FlightSpace's rule (known free, under the roof,
  clear of obstacles) for targets, RRT* and the travel-distance grid alike;
- only exact RRT* solutions are used, with the travel-distance grid's path as
  the fallback; a target neither reaches is blacklisted.
"""

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

from scage.geometry import compute_alignment_transforms, compute_visible_voxels, pose_difference
from scage.planner import FlightPlanner
from scage.space import FlightSpace


def _grid_keys(pts, res):
    """One int64 per point naming the nearest node of the res-spaced grid
    Wavemap's occupancy is sampled on (its query points sit at multiples of res)."""
    g = np.round(np.asarray(pts) / res).astype(np.int64) + (1 << 20)
    return (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]


def _inside(pts, box):
    box = np.asarray(box, float)
    return ((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[1]) & (pts[:, 1] >= box[2])
            & (pts[:, 1] <= box[3]) & (pts[:, 2] >= box[4]) & (pts[:, 2] <= box[5]))


class Target:
    """A candidate camera view. The fields live in `features`, which also
    holds the SCAGE ones (anchor, score, mass, ...; AnomalyManager.FEATURES)."""

    def __init__(self):
        self.features = {
            "3d_pos": None,  # camera position
            "gain": None,  # g(T), unknown volume it would see (m^3)
            "vd": None,  # optical axis in the world frame
            "valid_flag": True,
            "u_gain": None,  # gain after the visited-pose reduction
            "utility": None,  # Eq. 4
            "id": 0,  # unique within the manager
            "6d_pose": None,  # W_T_C
        }

    @property
    def pose6d(self):
        return self.features.get("6d_pose", None)

    @pose6d.setter
    def pose6d(self, val):
        if val.shape != (4, 4) and val is not None:
            raise ValueError("6D pose must be a 4x4 matrix.")
        self.features["6d_pose"] = val

    @property
    def id(self) -> int:
        return self.features["id"]

    @id.setter
    def id(self, val: int):
        self.features["id"] = val

    @property
    def pos3d(self):
        return self.features["3d_pos"]

    @pos3d.setter
    def pos3d(self, val):
        self.features["3d_pos"] = val

    @property
    def gain(self):
        return self.features["gain"]

    @gain.setter
    def gain(self, val):
        self.features["gain"] = val

    @property
    def view_direction(self):
        return self.features["vd"]

    @view_direction.setter
    def view_direction(self, val):
        self.features["vd"] = val

    @property
    def u_gain(self):
        return self.features["u_gain"]

    @u_gain.setter
    def u_gain(self, val):
        self.features["u_gain"] = val

    @property
    def utility(self):
        return self.features["utility"]

    @utility.setter
    def utility(self, val):
        self.features["utility"] = val

    @property
    def is_valid(self) -> bool:
        return self.features["valid_flag"]

    def set_invalid(self):
        self.features["valid_flag"] = False

    def set_valid(self):
        self.features["valid_flag"] = True

    def to_dict(self):
        """JSON-serializable dict (arrays become lists, Nones kept as None)."""

        def _list(x):
            return None if x is None else np.asarray(x, dtype=float).tolist()

        return {
            "id": self.id,
            "3d_pos": _list(self.pos3d),
            "gain": self.gain,
            "vd": _list(self.view_direction),
            "valid_flag": self.is_valid,
            "u_gain": self.u_gain,
            "utility": self.utility,
        }


class AnomalyManager:
    FEATURES = ("anchor", "score", "n_pts", "radius", "attempt", "mass", "n_vis", "members",
                "kind", "travel", "tour_rank")

    def __init__(self, params, log_level=20):
        p = params
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.frontiers = {}             # id -> Target
        self.robot_poses = {}           # id -> W_T_C
        self.occ_map = None
        self.free_map = None
        self.current_goal_pose = None
        # FrontierNet's id ranges: targets from 0, robot poses from 1000001
        self._next_target_id = 0
        self._next_robot_id = 1000001

        self.filter_bbox = p.get("filter_bbox", None)
        self.filter_max_vd_z = p.get("filter_max_vd_z", None)
        self.filter_min_gain = float(p.get("filter_min_gain", 0.1))
        self.max_planning_time = float(p.get("max_planning_time", 1.5))
        self.planning_algo = p.get("planning_algo", "rrtstar")
        self.v_tras_thre = float(p.get("visited_trans_threshold", 0.4))
        self.v_angl_thre = float(p.get("visited_angle_threshold", 0.4))
        self.v_gain_reduction_factor = float(p.get("visited_gain_reduction_factor", 1000))
        if p.get("render_K") is not None:
            self.render_K = np.array(p["render_K"], dtype=np.float32).reshape(3, 3)
        else:
            self.render_K = np.array([[60, 0, 64], [0, 60, 64], [0, 0, 1]], dtype=np.float32)
        self.render_H = int(p.get("render_H", 128))
        self.render_W = int(p.get("render_W", 128))
        self.render_d_range = float(p.get("render_depth_range", 3.5))
        self.voxel_size = float(p.get("voxel_size", 0.1))
        if self.filter_bbox is not None and len(self.filter_bbox) != 6:
            raise ValueError("filter_bbox must be [xmin, xmax, ymin, ymax, zmin, zmax].")

        sc = params["scage"]
        self.eps_dist = float(sc["eps_dist"])
        self.group_trans = float(sc["group_trans"])
        self.group_angle = float(sc["group_angle"])
        self.gain_voxel = float(sc.get("gain_voxel", self.voxel_size))
        self.gain_bbox = sc.get("gain_bbox")        # None: count unknown volume anywhere
        self.surface_obstacles = bool(sc.get("surface_obstacles", True))
        self.surface_map = None
        self.surface_voxel = float(sc["cloud_voxel"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._obstacles = None          # (points, splat radius) on the GPU, per map update
        self._depth_cache = {}          # obstacle depth per camera, per map update
        self._known = np.empty(0, np.int64)
        # poses the robot could not reach or gave up on; every Delta-Net run
        # would otherwise recreate the same target and the robot would retry it
        self.unreachable = []
        self.tour = []                  # frontier ids, in visiting order

        self.space = FlightSpace(
            params["bounds"], res=self.voxel_size, min_dist2occ=float(params["min_dist2occ"]),
            roof=bool(sc.get("indoor", True)), roof_nz=float(sc.get("roof_nz", 0.8)),
            roof_close=float(sc.get("roof_close", 0.6)), sky_z=float(sc.get("sky_z", 3.3)),
            trail_radius=float(sc.get("trail_radius", 0.3)),
            geo_step=float(sc.get("geo_step", 0.2)),
            segment_step=float(sc.get("motion_check_step", 0.05)))
        self.planner = FlightPlanner(params, self.space)
        self.plan_stats = dict(rrt=0, grid=0, none=0)
        # FrontierNet checks RRT* edges every 1% of the state space's extent,
        # ~0.3 m in a house - coarse enough to clip a corner between checks
        si = self.planner.ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(
            float(sc.get("motion_check_step", 0.05)) / si.getStateSpace().getMaximumExtent())

    def update_map(self, free_map=None, occ_map=None, surface=None, surface_nz=None,
                   trail=None):
        """Navigation map in, plus the observed surface from the Delta-Net map.

        Wavemap adds at most +0.4 log-odds per frame and FrontierNet calls a
        voxel occupied above 0.6, so a surface seen once is not yet occupied -
        and those first-look surfaces are exactly the ones next to the targets.
        With `surface_obstacles` the observed surface points join the planner's
        obstacles (clearance, line of sight), and they always join the
        occluders g(T) is rendered against. The known-volume count stays on
        Wavemap's own voxels. The surface normals and the camera trail build
        the flight space's roof map.
        """
        if free_map is not None:
            self.free_map = free_map
        if occ_map is not None:
            self.occ_map = occ_map
        if surface is not None:
            self.surface_map = surface
        obstacles = self.occ_map
        if self.surface_obstacles and self.surface_map is not None and len(self.surface_map):
            obstacles = self.surface_map if obstacles is None or not len(obstacles) \
                else np.vstack([obstacles, self.surface_map])
        self.planner.update_space(
            occ_vx=obstacles if obstacles is not None and len(obstacles) else None)
        self.space.update(self.free_map, self.planner._occ_kdt, surface, surface_nz, trail)
        known = [g for g in (self.free_map, self.occ_map) if g is not None and len(g)]
        self._known = np.unique(_grid_keys(np.vstack(known), self.voxel_size)) if known \
            else np.empty(0, np.int64)
        self._obstacles = None
        self._depth_cache = {}

    # ---- target set -------------------------------------------------------

    @property
    def valid_frontiers(self):
        return [ft for ft in self.frontiers.values() if ft.is_valid]

    def add_robot_poses(self, robot_poses):
        """-> the new poses' ids."""
        ids = []
        for pose in robot_poses:
            assert pose.shape == (4, 4), "Each robot pose must be a 4x4 matrix."
            rid = self._next_robot_id
            self._next_robot_id += 1
            self.robot_poses[rid] = pose
            ids.append(rid)
        return ids

    @staticmethod
    def target_pose(pos3d, view_direction):
        """W_T_C at pos3d with +Z along view_direction and +Y as close to
        straight down as it allows (CV convention)."""
        pos = np.asarray(pos3d, dtype=float).reshape(3)
        vd = np.asarray(view_direction, dtype=float).reshape(3)
        T = compute_alignment_transforms(
            origins=[pos],
            align_vec=vd,
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],  # CV convention: Y down approx
            appr_axis=[0, 1, 0],
        )[0]
        T = np.asarray(T, dtype=float)
        T[:3, 3] = pos
        return T

    def add_frontiers(self, frontiers):
        for ft in frontiers:
            ft.id = self._next_target_id
            self._next_target_id += 1
            self.frontiers[ft.id] = ft
            ft.pose6d = self.target_pose(ft.pos3d, ft.view_direction)

    def remove_frontiers(self, frontier_ids):
        for fid in frontier_ids:
            self.frontiers.pop(fid, None)

    def filter_frontiers(self):
        """Drop targets outside filter_bbox, with gain under filter_min_gain,
        pitched more than filter_max_vd_z, or too close to an obstacle."""
        bbox = self.filter_bbox
        min_gain = float(self.filter_min_gain)
        max_vd_z = self.filter_max_vd_z
        check_occ = self.occ_map is not None

        for fid, ft in list(self.frontiers.items()):
            if not ft.is_valid:
                continue
            if bbox is not None:
                x, y, z = map(float, ft.pos3d)
                if not (bbox[0] <= x <= bbox[1] and bbox[2] <= y <= bbox[3]
                        and bbox[4] <= z <= bbox[5]):
                    ft.set_invalid()
                    continue
            g = ft.u_gain if ft.u_gain is not None else ft.gain
            g = float(g if g is not None else 0.0)
            if g < min_gain:
                ft.set_invalid()
                continue
            if max_vd_z is not None and abs(float(ft.view_direction[2])) > max_vd_z:
                ft.set_invalid()
                continue
            if check_occ and self.planner.isoccupied(ft.pos3d):
                ft.set_invalid()
                continue

        self.remove_frontiers([fid for fid, ft in self.frontiers.items() if not ft.is_valid])

    def set_targets(self, targets):
        """Replace every target with this Delta-Net run's. Gain, clubbing and
        ranking follow in scage.selection."""
        self.remove_frontiers(list(self.frontiers.keys()))
        if self.unreachable and targets:
            dt, dr = pose_difference(np.stack([t.pose for t in targets]),
                                     np.stack(self.unreachable))
            blocked = ((dt < self.group_trans) & (dr < self.group_angle)).any(1)
            targets = [t for t, b in zip(targets, blocked) if not b]
        fts = []
        for t in targets:
            ft = Target()
            ft.pos3d = np.asarray(t.position, float)
            ft.view_direction = np.asarray(t.view_dir, float)
            ft.features.update(anchor=np.asarray(t.anchor, float), score=float(t.score),
                               n_pts=int(t.n_pts), radius=float(t.radius),
                               attempt=int(t.attempt), kind="cluster", members=1)
            ft.set_valid()
            fts.append(ft)
        if fts:
            self.add_frontiers(fts)

    def add_view(self, pose, **features):
        """A target that is not one cluster's own view (a clubbed one)."""
        ft = Target()
        ft.pos3d = np.asarray(pose[:3, 3], float)
        ft.view_direction = np.asarray(pose[:3, 2], float)
        ft.features.update(features)
        ft.set_valid()
        self.add_frontiers([ft])
        return ft

    @classmethod
    def target_dict(cls, ft):
        d = ft.to_dict()
        for k in cls.FEATURES:
            v = ft.features.get(k)
            d[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return d

    def blacklist(self, pose):
        self.unreachable.append(np.asarray(pose, float).copy())

    def blacklisted(self, pose):
        if not self.unreachable:
            return False
        dt, dr = pose_difference(np.asarray(pose, float)[None], np.stack(self.unreachable))
        return bool(((dt < self.group_trans) & (dr < self.group_angle)).any())

    # ---- Eq. 4 terms --------------------------------------------------------

    MAX_SPLAT_PX = 16

    def _obstacle_tensors(self):
        """Occupied voxels and observed surface points, with the radius each is
        splatted at: voxels half their width, surface points just wide enough
        to close the gaps of their 0.05 m grid."""
        if self._obstacles is None:
            pts, rad = [], []
            for g, r in ((self.occ_map, self.voxel_size / 2),
                         (self.surface_map, 0.75 * self.surface_voxel)):
                if g is not None and len(g):
                    pts.append(np.asarray(g, np.float32))
                    rad.append(np.full(len(g), r, np.float32))
            self._obstacles = (None, None) if not pts else tuple(
                torch.from_numpy(np.concatenate(a)).to(self.device) for a in (pts, rad))
        return self._obstacles

    @torch.no_grad()
    def render_obstacle_depth(self, C_T_W, chunk=8):
        """(B,4,4) world->camera -> (B,H,W) depth of the nearest obstacle, 0 where
        there is none.

        Every point is grown over its own image footprint, f * r / z pixels.
        FrontierNet's rasterize_voxel_depth instead grows every voxel by the
        footprint of the nearest voxel in the whole camera batch: one target
        close to a wall then blots out the view of every target rendered with
        it, and a target's gain depends on which others share its batch.
        """
        C_T_W = np.asarray(C_T_W, np.float32).reshape(-1, 4, 4)
        keys = [T.tobytes() for T in C_T_W]
        todo = [i for i, k in enumerate(keys) if k not in self._depth_cache]
        if todo:
            for i, d in zip(todo, self._render(C_T_W[todo], chunk)):
                self._depth_cache[keys[i]] = d
        return np.stack([self._depth_cache[k] for k in keys])

    def _render(self, C_T_W, chunk):
        B, H, W = len(C_T_W), self.render_H, self.render_W
        out = np.zeros((B, H, W), np.float32)
        pts, rad = self._obstacle_tensors()
        if pts is None:
            return out
        fx, fy = float(self.render_K[0, 0]), float(self.render_K[1, 1])
        cx, cy = float(self.render_K[0, 2]), float(self.render_K[1, 2])
        T = torch.from_numpy(C_T_W).to(self.device)
        for b0 in range(0, B, chunk):
            Tb = T[b0:b0 + chunk]
            nb = len(Tb)
            cam = torch.einsum("bij,nj->bni", Tb[:, :3, :3], pts) + Tb[:, None, :3, 3]
            z = cam[..., 2]
            bi, ni = ((z > 0.1) & (z < self.render_d_range + 1.0)).nonzero(as_tuple=True)
            z = z[bi, ni]
            u = (fx * cam[..., 0][bi, ni] / z + cx).round().long()
            v = (fy * cam[..., 1][bi, ni] / z + cy).round().long()
            keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            bi, ni, z, u, v = bi[keep], ni[keep], z[keep], u[keep], v[keep]
            r_px = (fx * rad[ni] / z).round().long().clamp(0, self.MAX_SPLAT_PX)
            flat = bi * (H * W) + v * W + u
            depth = torch.full((nb, H, W), float("inf"), device=self.device)
            for r in torch.unique(r_px).tolist():
                sel = r_px == r
                buf = torch.full((nb * H * W,), float("inf"), device=self.device)
                buf.scatter_reduce_(0, flat[sel], z[sel], reduce="amin", include_self=True)
                buf = buf.view(nb, 1, H, W)
                if r > 0:
                    buf = -F.max_pool2d(-buf, kernel_size=2 * r + 1, stride=1, padding=r)
                depth = torch.minimum(depth, buf[:, 0])
            depth[torch.isinf(depth)] = 0.0
            out[b0:b0 + nb] = depth.cpu().numpy()
        return out

    def is_known(self, pts):
        """True where Wavemap has the point's cell as free or occupied."""
        if len(pts) == 0 or len(self._known) == 0:
            return np.zeros(len(pts), bool)
        k = _grid_keys(pts, self.voxel_size)
        i = np.clip(np.searchsorted(self._known, k), 0, len(self._known) - 1)
        return self._known[i] == k

    def expected_gain(self, poses):
        """g(T): unknown volume (m^3) the target camera would see - inside its
        frustum, out to the sensor range, in front of the nearest obstacle, and
        not yet free or occupied in Wavemap. poses: (N,4,4) W_T_C.

        The frustum is sampled on a gain_voxel grid and each sample is looked
        up in Wavemap's cells. For fixed camera and grid settings, the number
        of frustum samples is bounded independently of map size. Overall
        computation still depends on map size, including obstacle rendering
        over the accumulated geometry."""
        poses = np.asarray(poses, float).reshape(-1, 4, 4)
        if len(poses) == 0:
            return np.zeros(0)
        C_T_W = np.linalg.inv(poses)
        depths = self.render_obstacle_depth(C_T_W)
        g = np.zeros(len(poses))
        for i, (T, depth) in enumerate(zip(C_T_W, depths)):
            seen, _ = compute_visible_voxels(K=self.render_K, T=T,
                                             img_size=(self.render_H, self.render_W),
                                             voxel_size=self.gain_voxel,
                                             max_depth=self.render_d_range,
                                             min_depth_map=depth)
            if self.gain_bbox is not None:
                seen = seen[_inside(seen, self.gain_bbox)]
            g[i] = (~self.is_known(seen)).sum() * self.gain_voxel ** 3
        return g

    def history_gain(self, poses, g):
        """FrontierNet's history term: a view the robot has already stood at,
        looking the same way, keeps ~no gain."""
        g = np.asarray(g, float).copy()
        if self.robot_poses and len(poses):
            robot = np.stack(list(self.robot_poses.values()))
            dt, dr = pose_difference(np.asarray(poses, float), robot)
            n_close = ((dt < self.v_tras_thre) & (dr < self.v_angl_thre)).sum(1)
            g = g - self.v_gain_reduction_factor * n_close
        return np.maximum(g, 1e-4)

    def gain_adjustment(self, w_history_path=True, w_map=True):
        """gain = g(T) against the current map; u_gain additionally drops to
        ~0 for targets the robot has already stood at, looking the same way."""
        fts = self.valid_frontiers
        if not fts:
            return
        poses = np.stack([ft.pose6d for ft in fts])
        if w_map and self.occ_map is not None and self.free_map is not None:
            g = self.expected_gain(poses)
            for ft, gi in zip(fts, g):
                ft.gain = float(gi)
        g = np.array([ft.gain or 0.0 for ft in fts])
        u = self.history_gain(poses, g) if w_history_path else np.maximum(g, 1e-4)
        for ft, ui in zip(fts, u):
            ft.u_gain = float(ui)

    @torch.no_grad()
    def visible(self, poses, pts, see_range, tol=0.2, probes=None, chunk=16):
        """(N,4,4) W_T_C, (M,3) points -> (N,M) bool: in the camera's image,
        no farther than see_range, and not behind the nearest obstacle (the
        same per-point splat render g(T) uses; tol absorbs the splat's
        footprint on the surface the point itself lies on).

        probes (M,3): a second point per point that must pass the same test.
        For an anomalous point that is a step along Delta-Net's bearing, into
        the unobserved space the anomaly points at: a view that sees the rim
        but not past it does not resolve it."""
        poses = np.asarray(poses, float).reshape(-1, 4, 4)
        out = np.zeros((len(poses), len(pts)), bool)
        if not len(poses) or not len(pts):
            return out
        C_T_W = np.linalg.inv(poses).astype(np.float32)
        depth = torch.from_numpy(self.render_obstacle_depth(C_T_W)).to(self.device)
        T = torch.from_numpy(C_T_W).to(self.device)
        sets = [torch.from_numpy(np.asarray(q, np.float32)).to(self.device)
                for q in ((pts,) if probes is None else (pts, probes))]
        K = self.render_K
        H, W = self.render_H, self.render_W
        for b0 in range(0, len(poses), chunk):
            Tb, Db = T[b0:b0 + chunk], depth[b0:b0 + chunk]
            ok = None
            for P in sets:
                cam = torch.einsum("bij,nj->bni", Tb[:, :3, :3], P) + Tb[:, None, :3, 3]
                z = cam[..., 2]
                zs = z.clamp(min=1e-6)
                u = (K[0, 0] * cam[..., 0] / zs + K[0, 2]).round().long()
                v = (K[1, 1] * cam[..., 1] / zs + K[1, 2]).round().long()
                seen = (z > 0.1) & (z <= see_range) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                b = torch.arange(len(Tb), device=self.device)[:, None].expand_as(u)
                d = Db[b, v.clamp(0, H - 1), u.clamp(0, W - 1)]
                seen &= (d <= 0) | (z <= d + tol)
                ok = seen if ok is None else ok & seen
            out[b0:b0 + len(Tb)] = ok.cpu().numpy()
        return out

    # ---- planning -----------------------------------------------------------

    def plan_to(self, goal_pose, current_pose, field=None):
        """Dense list of W_T_C from the current pose to goal_pose, or None.

        RRT* first (exact solutions only); if it finds none in its time
        budget, the travel-distance grid's path, shortcut. -> (path, how)"""
        pl = self.planner
        path, how = None, None
        if pl.update_start_goal(current_pose, goal_pose) and \
                pl.solve(time_limit=self.max_planning_time, method=self.planning_algo):
            pl.interpolate_path()
            path, how = pl.get_solution_path(return_type="mat"), "rrt"
        else:
            if field is None:
                field = self.space.field(current_pose[:3, 3])
            wps = self.space.path(field, goal_pose[:3, 3])
            if wps is not None:
                pl.set_start_goal(current_pose, goal_pose)
                pl.interpolate_path(external_waypoints=wps)
                path, how = pl.get_solution_path(return_type="mat"), "grid"
        self.plan_stats[how or "none"] += 1
        if path:
            self.current_goal_pose = np.asarray(goal_pose, float)
        return path, how

    @staticmethod
    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    def write_to_file(self, file_path):
        """Append one JSON line: targets with their anomaly fields, robot poses, goal."""
        entry = {
            "targets": [self.target_dict(ft) for ft in self.valid_frontiers],
            "robot_poses": {int(r): p.tolist() for r, p in self.robot_poses.items()},
            "current_goal_pose": None if self.current_goal_pose is None
            else self.current_goal_pose.tolist(),
        }
        with open(file_path, "a") as f:
            f.write(json.dumps(entry, default=self._json_default) + "\n")
