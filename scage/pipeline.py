"""The SCAGE loop body, headless: frames in, ranked safe targets out.

The Open3D app (`scage_explore.py`) and the tests both drive this, so what is
tested is what runs. Two maps are kept:

  navigation  Wavemap at `voxel_size` (0.1 m), exactly as FrontierNet - free
              and occupied grids for the planner and for gain raycasting
  Delta-Net   PointMap at `scage.cloud_voxel` (0.05 m) - the surface the
              anomaly network scores

Both are fed by the same `mapper.insert_depth_to_buffer` call, on every frame
the robot captures. Delta-Net runs only every few steps (`detect`); in between
both maps keep growing, and the navigation map and flight space are refreshed
each step (`update_nav_map`).
"""

import time

import numpy as np
from scipy.spatial import cKDTree

from scage.anomaly import AnomalyScorer, minmax
from scage.clustering import find_clusters
from scage.manager import AnomalyManager
from scage.point_map import DualMapper, PointMap
from scage.selection import select_views
from scage.viewpoints import ViewpointGenerator
from scage.wavemap import WaveMapper


class ScagePipeline:
    def __init__(self, config, K, H, W, device="cuda", log_level=20):
        self.config = config
        sc = self.sc = config["scage"]
        K = np.asarray(K, float)
        self.fov = 2 * np.arctan(W / (2 * K[0, 0]))
        voxel = float(config["voxel_size"])

        self.points = PointMap(K, sc["cloud_voxel"], sc["min_range"], sc["max_range"])
        wave = WaveMapper(params={
            "min_cell_width": voxel / 2.0, "width": W, "height": H,
            "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
            "min_range": 0.05, "max_range": float(config["depth_range"]),
            "resolution": voxel,
        })
        self.scorer = AnomalyScorer(overlap=sc["overlap"], batch=sc["batch"],
                                    agg=sc["agg"], device=device)
        self.manager = AnomalyManager(params=config, log_level=log_level)
        self.mapper = DualMapper(wave, self.points, self.manager.space)
        self.viewpoints = ViewpointGenerator(
            self.manager.space, self.fov, d_min=sc["d_min"], normal_k=sc["normal_k"],
            max_vd_z=config["filter_max_vd_z"], target_clearance=sc["target_clearance"])
        # a cluster wider than this would need a standoff beyond d_max
        self.max_cluster_radius = sc["d_max"] * np.tan(self.fov / 2)
        self.last = {}
        self.scored = False
        self.frames_looked = 0          # frames already counted into PointMap.looked

    def observe(self, depth, W_T_C, rgb=None):
        self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C, rgb=rgb)

    def update_nav_map(self):
        self.mapper.integrate_from_buffer()
        self.mapper.interpolate_occupancy_grid()
        og = self.mapper.get_occupancy_grid()
        trail = np.stack([T[:3, 3] for T in self.mapper.frames]) if self.mapper.frames \
            else None
        self.manager.update_map(free_map=og["free"], occ_map=og["occupied"],
                                surface=self.points.xyz, surface_nz=self.points.nz,
                                trail=trail)

    def affected(self):
        """Points within incremental_radius of a point added since the last scoring."""
        pm = self.points
        if not pm.dirty.any():
            return np.zeros(len(pm), bool)
        r = float(self.sc.get("incremental_radius", 2.0))
        d, _ = cKDTree(pm.xyz[pm.dirty]).query(pm.xyz, distance_upper_bound=r, workers=-1)
        return np.isfinite(d)

    def anomaly_scores(self, incremental=None):
        """Score the point map and cache the result on it.

        Incremental: patches are seeded only where new points arrived (within
        incremental_radius of them) and every point those patches cover takes
        the new result; points no new patch reaches keep their cached score -
        nothing near them changed, so neither would their patch. The paper
        likewise skips re-processing settled neighbourhoods.
        """
        pm = self.points
        if incremental is None:
            incremental = bool(self.sc.get("incremental", False))
        if incremental and self.scored and len(pm):
            s, b, cov = self.scorer.score(pm.xyz, seed_mask=self.affected())
            upd = cov > 0
            pm.score[upd], pm.direction[upd] = s[upd], b[upd]
            pm.covered |= upd
        else:
            pm.score[:], pm.direction[:], cov = self.scorer.score(pm.xyz)
            pm.covered = cov > 0
        pm.dirty[:] = False
        self.scored = True
        if self.sc["score_norm"] == "minmax":
            return minmax(pm.score, pm.covered)
        return pm.score.copy()

    def detect(self):
        """Score the map, cluster, place safe targets, hand them to the manager."""
        t = [time.time()]
        self.update_nav_map()                 # targets are validated against the newest map
        t.append(time.time())
        s = self.anomaly_scores()
        t.append(time.time())
        pm, sc = self.points, self.sc
        clusters = find_clusters(pm.xyz, s, sc["tau"], sc["cluster_radius"],
                                 sc["min_cluster_pts"], self.max_cluster_radius)
        self.last = dict(score=s)
        self.count_looks()
        targets = self.viewpoints.generate(pm.xyz, s, pm.eye, pm.eyes, clusters)
        t.append(time.time())
        self.manager.set_targets(targets)
        t.append(time.time())
        self.last = dict(score=s, clusters=clusters, targets=targets)
        self.selected = False
        rep = self.viewpoints.last_report
        return dict(points=len(pm), anomalous=int((s > sc["tau"]).sum()),
                    clusters=len(clusters), targets=len(targets),
                    primary_ok=rep.get("primary_ok", 0),
                    primary_failures=rep.get("primary_failures", {}),
                    t_map=t[1] - t[0], t_score=t[2] - t[1], t_view=t[3] - t[2],
                    t_gain=t[4] - t[3])

    def anomalous(self, raw=False):
        """The last scoring's anomalous points: (xyz, weight, probe, rows).

        The probe is a step of probe_dist along Delta-Net's bearing, into the
        unobserved space the point borders. The weight is the score, halved
        (look_decay) for every frame that has already seen the point and past
        it while it stayed anomalous: Delta-Net sees only points, so a physical
        edge - a table rim, a door frame - looks like an observation boundary
        and never resolves. Looking again would not change that. raw: the plain
        score."""
        s = self.last.get("score")
        if s is None:
            return np.empty((0, 3)), np.empty(0), np.empty((0, 3)), np.empty(0, int)
        rows = np.flatnonzero(s > self.sc["tau"])
        xyz = self.points.xyz[rows].astype(float)
        b = self.points.direction[rows].astype(float)
        n = np.linalg.norm(b, axis=1, keepdims=True)
        b = np.where(n > 1e-9, b / np.maximum(n, 1e-9), 0.0)
        w = s[rows].astype(float)
        if not raw:
            w = w * float(self.sc["look_decay"]) ** np.minimum(self.points.looked[rows], 16)
        return xyz, w, xyz + float(self.sc["probe_dist"]) * b, rows

    def count_looks(self):
        """Frames captured since the last scoring that saw an anomalous point
        and past it: one more look each (PointMap.looked)."""
        frames = self.mapper.frames[self.frames_looked:]
        self.frames_looked = len(self.mapper.frames)
        xyz, _, probe, rows = self.anomalous(raw=True)
        if not frames or not len(rows):
            return
        V = self.manager.visible(np.stack(frames), xyz, float(self.sc["see_range"]),
                                 float(self.sc.get("vis_tol", 0.2)), probe)
        self.points.looked[rows] += V.sum(0).astype(np.int16)

    def select(self, robot_pos):
        """Club this run's targets into views and pick the tour (scage.selection)."""
        t = time.time()
        A_xyz, A_s, A_probe, _ = self.anomalous()
        st = select_views(self.manager, lambda P: self.viewpoints.check(P), robot_pos,
                          A_xyz, A_s, A_probe, self.sc)
        st["t_select"] = time.time() - t
        self.selected = True
        return st

    def view_mass(self, pose):
        """Anomaly mass (m^2) a camera at pose would resolve now."""
        A_xyz, A_s, A_probe, _ = self.anomalous()
        if not len(A_xyz):
            return 0.0
        vis = self.manager.visible(np.asarray(pose, float)[None], A_xyz,
                                   float(self.sc["see_range"]), float(self.sc.get("vis_tol", 0.2)),
                                   A_probe)[0]
        return float(A_s[vis].sum() * float(self.sc["cloud_voxel"]) ** 2)
