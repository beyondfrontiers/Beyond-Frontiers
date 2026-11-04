"""One safe 6-DoF camera target per anomaly cluster (paper Sec. III-B, Eq. 3).

For cluster C_k the camera goes at  P = a_k + d * n_k,  looking at a_k, where
  a_k  anchor, the cluster centroid
  n_k  surface normal at the anchor, turned toward the cameras that observed
       the cluster, so P lands on the observed - free - side
  d    = max(d_min, r_k / tan(fov / 2)), far enough that the whole cluster
       (radius r_k) fits in the field of view.

The pose is then made flyable: height clamped into the planning box, and moved
back horizontally if the view would be steeper than FrontierNet's
`filter_max_vd_z` allows. It must then pass every safety check below, against
the navigation map (Wavemap). If it does not, a ring of nearby alternatives is
tried - other bearings around the anchor, nearer and farther - and the first
safe one is kept. A cluster with no safe pose gets no target.

Safety, in order:
  1. the flight space's own rule (scage.space): inside the planning bounds, in
     a known-free Wavemap cell, under the roof (not outside the building) and
     > min_dist2occ from obstacles - so every target is also a legal RRT* goal
  2. >= target_clearance from obstacles (stricter than the planner's path
     clearance: the robot stops and looks from here)
  3. a ball of that radius around it is known free, not unknown - no
     unobserved obstacle right next to the target
  4. line of sight to the anchor is not blocked by an obstacle
Obstacles are the planner's: Wavemap's occupied voxels, plus the observed
surface points when `surface_obstacles` is on (see AnomalyManager.update_map).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from scage.geometry import compute_alignment_transforms


def standoff(radius, fov, d_min):
    """Eq. 3: nearest distance at which a cluster of this radius fits the FOV."""
    return max(d_min, radius / np.tan(fov / 2.0))


def camera_pose(position, view_dir):
    """W_T_C, CV camera (+z forward, +y down), the same construction
    AnomalyManager.target_pose uses."""
    T = compute_alignment_transforms(
        origins=[np.asarray(position, float)], align_vec=np.asarray(view_dir, float),
        align_axis=[0, 0, 1], appr_vec=[0, 0, -1], appr_axis=[0, 1, 0])[0]
    T[:3, 3] = position
    return T


def look_around(W_T_C, steps=8, pitch_deg=30.0):
    """A full turn in place, pitched up, then level again: W_T_C poses from the
    current one. A level camera never sees the ceiling within ~2 m of itself,
    and the flight space's roof map needs it (scage/space.py)."""
    pos, z = W_T_C[:3, 3], W_T_C[:3, 2]
    yaw0 = np.arctan2(z[1], z[0])
    p = np.radians(pitch_deg)
    path = [np.asarray(W_T_C, float)]
    for k in range(1, steps + 1):
        a = yaw0 + 2 * np.pi * k / steps
        path.append(camera_pose(pos, [np.cos(p) * np.cos(a), np.cos(p) * np.sin(a), np.sin(p)]))
    path.append(camera_pose(pos, [np.cos(yaw0), np.sin(yaw0), 0.0]))
    return path


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def _sphere_offsets(radius):
    """26 compass directions at the full and half radius."""
    g = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                  if (i, j, k) != (0, 0, 0)], float)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return np.concatenate([g * radius, g * radius * 0.5])


@dataclass
class Target:
    position: np.ndarray      # camera centre, world
    view_dir: np.ndarray      # unit optical axis, world
    anchor: np.ndarray        # a_k
    score: float              # s_k, mean anomaly score over the cluster
    n_pts: int
    radius: float             # r_k
    standoff: float           # d from Eq. 3, before any fallback scaling
    rows: np.ndarray          # cluster rows in the point map
    attempt: int              # 0 = the paper's pose, >0 = a fallback from the ring

    @property
    def pose(self):
        return camera_pose(self.position, self.view_dir)


class ViewpointGenerator:
    YAW_STEPS = (0, 15, -15, 30, -30, 45, -45)     # degrees around the anchor
    DIST_SCALES = (1.0, 0.75, 1.25)

    def __init__(self, space, fov, d_min=1.0, normal_k=150, max_vd_z=0.65,
                 target_clearance=0.3, free_frac=0.9, los_step=0.05, los_stop=0.25,
                 los_clearance=0.07, z_margin=0.1):
        self.space = space
        self.bounds = space.bounds
        self.fov = float(fov)
        self.d_min = float(d_min)
        self.normal_k = int(normal_k)
        self.max_vd_z = float(max_vd_z)
        self.target_clearance = float(target_clearance)
        self.free_frac = float(free_frac)
        self.los_step, self.los_stop, self.los_clearance = los_step, los_stop, los_clearance
        self.z_lo = self.bounds[2, 0] + z_margin
        self.z_hi = self.bounds[2, 1] - z_margin
        # horizontal run needed per metre of height difference to respect max_vd_z
        self._run_per_rise = np.sqrt(1 - self.max_vd_z ** 2) / self.max_vd_z
        self._ball = _sphere_offsets(self.target_clearance)
        self.last_report = {}

    # ---- safety -----------------------------------------------------------

    def check(self, P, a=None):
        """-> None if safe, else the name of the first failed check. Without an
        anchor a, line of sight is not checked (a clubbed view looks at several
        clusters; what it sees is raycast in scage.selection instead)."""
        P = np.asarray(P, float)
        why = self.space.reason(P)
        if why is not None:
            return why
        occ = self.space.tree
        if occ is not None and occ.query(P)[0] < self.target_clearance:
            return "clearance"
        if self.space.is_free(P + self._ball).mean() < self.free_frac:
            return "unknown_nearby"
        if occ is not None and a is not None:
            seg = a - P
            length = np.linalg.norm(seg)
            ts = np.arange(0.0, max(length - self.los_stop, 0.0), self.los_step)
            if len(ts) and (occ.query(P + np.outer(ts / length, seg))[0]
                            < self.los_clearance).any():
                return "line_of_sight"
        return None

    # ---- geometry ---------------------------------------------------------

    def _normal(self, tree, xyz, a, eyes):
        """PCA normal of the surface around the anchor, turned toward the eyes.

        Uses all map points near the anchor, not just the anomalous ones - a
        rim of anomalous points is a line, and a line has no normal."""
        k = min(self.normal_k, len(xyz))
        _, nb = tree.query(a, k=k)
        nb_pts = xyz[np.atleast_1d(nb)]
        w, v = np.linalg.eigh(np.cov((nb_pts - nb_pts.mean(0)).T))
        n = v[:, 0]
        toward = eyes.mean(0) - a
        return n if n @ toward >= 0 else -n, toward

    def _primary(self, a, n, toward, d):
        """P = a + d n, then made flyable. -> (azimuth, horizontal run, height)."""
        P = a + d * n
        z = float(np.clip(P[2], self.z_lo, self.z_hi))
        horiz = P[:2] - a[:2]
        run = np.linalg.norm(horiz)
        need = abs(z - a[2]) * self._run_per_rise + 0.05      # steepest allowed view
        need = max(need, np.sqrt(max(self.d_min ** 2 - (z - a[2]) ** 2, 0.0)))
        if run < 1e-3:                                     # normal is vertical
            horiz = toward[:2] if np.linalg.norm(toward[:2]) > 1e-3 else np.array([1.0, 0.0])
        run = max(run, need)
        return np.arctan2(horiz[1], horiz[0]), run, z, need

    def generate(self, xyz, score, eye_ids, eyes, clusters):
        """-> list of Target, one per cluster that has a safe pose."""
        tree = cKDTree(xyz)
        targets, fails = [], {}
        for rows in clusters:
            pts = xyz[rows]
            a = pts.mean(0)
            r = float(np.linalg.norm(pts - a, axis=1).max())
            d = standoff(r, self.fov, self.d_min)
            n, toward = self._normal(tree, xyz, a, eyes[np.unique(eye_ids[rows])])
            phi0, run0, z, need = self._primary(a, n, toward, d)

            found = None
            for attempt, (scale, yaw) in enumerate(
                    (s, y) for s in self.DIST_SCALES for y in self.YAW_STEPS):
                phi = phi0 + np.deg2rad(yaw)
                run = max(run0 * scale, need)
                P = np.array([a[0] + run * np.cos(phi), a[1] + run * np.sin(phi), z])
                why = self.check(P, a)
                if why is None:
                    found = (attempt, P)
                    break
                if attempt == 0:
                    fails[why] = fails.get(why, 0) + 1
            if found is None:
                continue
            attempt, P = found
            targets.append(Target(
                position=P, view_dir=_unit(a - P), anchor=a,
                score=float(score[rows].mean()), n_pts=len(rows), radius=r, standoff=d,
                rows=rows, attempt=attempt))
        self.last_report = dict(clusters=len(clusters), targets=len(targets),
                                primary_ok=sum(t.attempt == 0 for t in targets),
                                primary_failures=fails)
        return targets
