"""Where the camera may fly: one validity rule for targets, RRT* states, travel
distance and the per-step re-check.

A position p is valid when
  1. it is inside the planning bounds;
  2. the Wavemap cell containing it is known free. This is an exact lookup on
     the 0.1 m lattice Wavemap is sampled on. FrontierNet's test, "within
     0.075 m of a free cell centre", leaves out the corners of every free cell
     (up to 0.087 m away). RRT* edges and path poses then fall into those gaps;
  3. it is under the roof (below);
  4. it is more than min_dist2occ from every obstacle: Wavemap's occupied
     voxels plus the observed surface points.

The roof. Free space alone does not separate inside from outside: depth rays
leave through doors, windows and holes in the scan and carve free space out
there, and the robot followed them out. What the building has and the outside
does not is a ceiling. The roof map is a 2D grid of the cells a ceiling has been
seen over: points of the Delta-Net map whose normal faces down (n_z < -roof_nz),
above the highest flight altitude.

A camera looking level only sees the ceiling from ~2 m away, so the ceiling
right above the robot, just past a doorway, or under a lintel is not observed
yet. Three things cover that:
  - gaps narrower than 2 * roof_close are closed (morphological closing, which
    never grows a region past its outline);
  - unobserved patches enclosed by observed ceiling are filled, unless open sky
    was seen over them. A courtyard or patio enclosed by the building is such a
    patch: from inside, depth rays that hit nothing ("no return") leave through
    its openings and climb past the highest ceiling (sky_z). The columns a
    no-return ray crosses up there are marked sky (`observe_sky`). Filling
    every enclosed patch instead roofs over most of the real outdoor area in
    scene 00876, the patio the robot left through included;
  - the robot's own trail counts as roofed.
The result is shrunk by one cell (eaves, door thresholds) and the trail added
back.

This models a UAV: no floor or ground-contact requirement, so it may hover over
furniture. It assumes one storey: the roof is any ceiling above the flight band.
Several floors would need one roof map per storey.

Travel distance. A 26-connected graph over the valid lattice points every
geo_step (0.2 m) gives Dijkstra distances, so a target behind a wall is as far
away as the way round it, and a target in free space that is not connected is
infinitely far. Its predecessor tree also gives a path to fall back on when
RRT* finds none in its time budget.
"""

import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def _disk(radius_cells):
    r = int(np.ceil(radius_cells))
    i, j = np.mgrid[-r:r + 1, -r:r + 1]
    return i ** 2 + j ** 2 <= radius_cells ** 2 + 1e-9


_OFFSETS = np.array([o for o in np.ndindex(3, 3, 3)]) - 1
_OFFSETS = _OFFSETS[[tuple(o) > (0, 0, 0) for o in _OFFSETS]]    # 13 forward neighbours


class Field:
    """Dijkstra from one point: distance and predecessor per graph node."""

    def __init__(self, src, node, dist, pred):
        self.src, self.node, self.dist, self.pred = src, node, dist, pred


class FlightSpace:
    SKY_STRIDE = 4          # trace every 4th no-return pixel in each image axis
    SKY_BAND = 2.0          # mark columns while the ray is this far above sky_z
    SKY_RANGE = 30.0        # ... and no farther than this from the camera
    MAP_MARGIN = 15.0       # the 2D roof / sky maps reach this far past the bounds

    def __init__(self, bounds, res=0.1, min_dist2occ=0.2, roof=True, roof_nz=0.8,
                 roof_close=0.6, roof_min_z=None, sky_z=3.3, trail_radius=0.3,
                 geo_step=0.2, attach_radius=0.45, segment_step=0.05, margin=0.5):
        self.bounds = np.asarray(bounds, float).reshape(3, 2)
        self.res = float(res)
        self.min_dist2occ = float(min_dist2occ)
        self.use_roof = bool(roof)
        self.roof_nz = float(roof_nz)
        self.roof_close = float(roof_close)
        self.roof_min_z = float(self.bounds[2, 1] if roof_min_z is None else roof_min_z)
        self.sky_z = float(sky_z)
        self.trail_radius = float(trail_radius)
        self.geo_cells = max(int(round(geo_step / res)), 1)
        self.attach_radius = float(attach_radius)
        self.segment_step = float(segment_step)

        # the lattice: bounds plus a margin, so checks just outside the flight
        # box (the free ball around a target) still see Wavemap's free cells
        self.n0 = self._idx(self.bounds[:, 0] - margin)
        n1 = self._idx(self.bounds[:, 1] + margin)
        self.shape = tuple(int(v) for v in n1 - self.n0 + 1)
        ax = [(self.n0[d] + np.arange(self.shape[d])) * self.res for d in range(3)]
        inb = [(a >= self.bounds[d, 0] - 1e-6) & (a <= self.bounds[d, 1] + 1e-6)
               for d, a in enumerate(ax)]
        self.inb = inb[0][:, None, None] & inb[1][None, :, None] & inb[2][None, None, :]

        # the 2D maps: a wide margin, so a courtyard's sky marks are kept even
        # where they fall outside the flight box
        self.m0 = self._idx(self.bounds[:2, 0] - self.MAP_MARGIN)
        self.mshape = tuple(int(v) for v in
                            self._idx(self.bounds[:2, 1] + self.MAP_MARGIN) - self.m0 + 1)
        self.sky = np.zeros(self.mshape, bool)
        self.roof_wide = np.ones(self.mshape, bool)

        self.free = np.zeros(self.shape, bool)
        self.roof = np.ones(self.shape[:2], bool)
        self.ok = np.zeros(self.shape, bool)       # bounds & free & roof
        self.tree = None
        self._graph = None

    # ---- lattice ------------------------------------------------------------

    def _idx(self, p):
        return np.floor(np.asarray(p, float) / self.res + 0.5).astype(np.int64)

    def _cells(self, P):
        """-> (lattice index (N,3) relative to the grid, inside-grid mask)"""
        idx = self._idx(np.atleast_2d(P)) - self.n0
        inr = ((idx >= 0) & (idx < np.array(self.shape))).all(1)
        return idx, inr

    def centres(self, idx):
        return (np.asarray(idx) + self.n0) * self.res

    # ---- update ------------------------------------------------------------

    def update(self, free_pts, tree, surface=None, surface_nz=None, trail=None):
        """free_pts: Wavemap free cell centres; tree: cKDTree of the obstacles;
        surface / surface_nz: the Delta-Net map and its normals' z; trail:
        camera positions so far."""
        self.free[:] = False
        if free_pts is not None and len(free_pts):
            idx, inr = self._cells(free_pts)
            self.free[tuple(idx[inr].T)] = True
        self.tree = tree
        if self.use_roof:
            self.roof = self._roof_map(surface, surface_nz, trail)
        self.ok = self.free & self.inb & self.roof[:, :, None]
        self._graph = None

    def ceiling_points(self, surface, surface_nz):
        if surface is None or not len(surface):
            return np.empty((0, 3))
        nz = np.nan_to_num(np.asarray(surface_nz, float), nan=0.0)
        return np.asarray(surface)[(nz < -self.roof_nz) & (surface[:, 2] > self.roof_min_z)]

    def _mark2d(self, img, xy):
        ij = self._idx(np.asarray(xy, float)[:, :2]) - self.m0
        keep = ((ij >= 0) & (ij < np.array(img.shape))).all(1)
        img[tuple(ij[keep].T)] = True

    def add_sky(self, xy):
        """Mark the columns over these points as open sky."""
        self._mark2d(self.sky, xy)

    def observe_sky(self, depth, W_T_C, K):
        """Trace the no-return pixels of a depth image (depth 0: the ray hit
        nothing) and mark the columns they cross between sky_z and
        sky_z + SKY_BAND."""
        depth = np.asarray(depth)
        st = self.SKY_STRIDE
        v, u = np.nonzero(depth[::st, ::st] == 0)
        if not len(u):
            return
        u, v = u * st, v * st
        K, T = np.asarray(K, float), np.asarray(W_T_C, float)
        d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))], 1)
        d = d @ T[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = T[:3, 3]
        d = d[d[:, 2] > 1e-3]
        t0 = (self.sky_z - c[2]) / d[:, 2]
        t1 = np.minimum((self.sky_z + self.SKY_BAND - c[2]) / d[:, 2], self.SKY_RANGE)
        ok = (t0 < t1) & (t0 > 0)
        d, t0, t1 = d[ok], t0[ok], t1[ok]
        if not len(d):
            return
        n = int(np.ceil(np.max(np.linalg.norm(d[:, :2], axis=1) * (t1 - t0)) / self.res)) + 1
        tt = t0[:, None] + (t1 - t0)[:, None] * np.linspace(0.0, 1.0, max(n, 2))[None]
        self.add_sky((c[None, None, :2] + tt[..., None] * d[:, None, :2]).reshape(-1, 2))

    def _roof_map(self, surface, surface_nz, trail):
        raw = np.zeros(self.mshape, bool)
        ceil = self.ceiling_points(surface, surface_nz)
        if len(ceil):
            self._mark2d(raw, ceil)
        trail_img = np.zeros(self.mshape, bool)
        if trail is not None and len(trail):
            self._mark2d(trail_img, trail)
            trail_img = ndimage.binary_dilation(trail_img, _disk(self.trail_radius / self.res))
        img = raw | trail_img
        if self.roof_close > 0:
            disk = _disk(self.roof_close / self.res)
            img = ndimage.binary_erosion(ndimage.binary_dilation(img, disk), disk)
        lab, n = ndimage.label(~img)
        if n:
            border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            skyl = np.unique(lab[self.sky & ~img])
            fill = np.setdiff1d(np.arange(1, n + 1), np.union1d(border, skyl))
            img |= np.isin(lab, fill)
        img = ndimage.binary_erosion(img) | trail_img
        self.roof_wide = img
        o = self.n0[:2] - self.m0
        return img[o[0]:o[0] + self.shape[0], o[1]:o[1] + self.shape[1]]

    # ---- validity --------------------------------------------------------

    def valid(self, p):
        """Scalar fast path, for OMPL's per-state callback."""
        r = self.res
        i = int(math.floor(p[0] / r + 0.5)) - self.n0[0]
        j = int(math.floor(p[1] / r + 0.5)) - self.n0[1]
        k = int(math.floor(p[2] / r + 0.5)) - self.n0[2]
        nx, ny, nz = self.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz) or not self.ok[i, j, k]:
            return False
        return self.tree is None or self.tree.query(p)[0] > self.min_dist2occ

    def valid_many(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        idx, inr = self._cells(P)
        ok = np.zeros(len(P), bool)
        ok[inr] = self.ok[tuple(idx[inr].T)]
        if self.tree is not None and ok.any():
            d, _ = self.tree.query(P[ok], distance_upper_bound=self.min_dist2occ)
            ok[ok] = d > self.min_dist2occ
        return ok

    def reason(self, p):
        """None if valid, else the first rule p breaks."""
        p = np.asarray(p, float)
        if not ((p >= self.bounds[:, 0]) & (p <= self.bounds[:, 1])).all():
            return "bounds"
        idx, inr = self._cells(p)
        if not inr[0] or not self.free[tuple(idx[0])]:
            return "unknown"
        if not self.roof[idx[0, 0], idx[0, 1]]:
            return "outside"
        if self.tree is not None and self.tree.query(p)[0] <= self.min_dist2occ:
            return "clearance"
        return None

    def flyable(self, P):
        """In bounds, known free and under the roof - everything but clearance."""
        idx, inr = self._cells(P)
        out = np.zeros(len(idx), bool)
        out[inr] = self.ok[tuple(idx[inr].T)]
        return out

    def is_free(self, P):
        """Wavemap knows the cell as free (no bounds, roof or clearance)."""
        idx, inr = self._cells(P)
        out = np.zeros(len(idx), bool)
        out[inr] = self.free[tuple(idx[inr].T)]
        return out

    def indoors(self, P):
        idx, inr = self._cells(P)
        out = np.zeros(len(idx), bool)
        out[inr] = self.roof[idx[inr, 0], idx[inr, 1]]
        return out

    def clearance(self, P):
        if self.tree is None:
            return np.full(len(np.atleast_2d(P)), np.inf)
        return self.tree.query(np.atleast_2d(P))[0]

    def segment_valid(self, a, b, skip_start=0.0):
        """Every point of the straight segment a -> b, every segment_step, is
        valid - except within skip_start of a (to leave a spot the newest map
        has just put too close to a surface)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = np.linalg.norm(b - a)
        n = max(int(np.ceil(L / self.segment_step)), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        if L > 0:
            t = t[t * L >= skip_start]
        return bool(self.valid_many(a + np.outer(t, b - a)).all()) if len(t) else True

    def nearest_valid(self, p, radius=None):
        """Nearest valid lattice point within radius, or None."""
        radius = self.attach_radius if radius is None else radius
        r = int(np.ceil(radius / self.res))
        off = np.argwhere(np.ones((2 * r + 1,) * 3, bool)) - r
        cand = self.centres(self._idx(p)[None] - self.n0 + off)
        d = np.linalg.norm(cand - p, axis=1)
        cand, d = cand[d <= radius], d[d <= radius]
        ok = self.valid_many(cand)
        if not ok.any():
            return None
        return cand[ok][np.argmin(d[ok])]

    # ---- travel distance ---------------------------------------------------

    def graph(self):
        """Valid lattice points every geo_cells, linked to their 26 neighbours
        when the lattice points in between are valid too. Built once per map."""
        if self._graph is not None:
            return self._graph
        s = self.geo_cells
        clear = self.ok.copy()
        if self.tree is not None and clear.any():
            pts = self.centres(np.argwhere(clear))
            d, _ = self.tree.query(pts, distance_upper_bound=self.min_dist2occ, workers=-1)
            clear[clear] = d > self.min_dist2occ
        sub = clear[::s, ::s, ::s]
        cu = np.argwhere(sub)
        node = -np.ones(sub.shape, np.int64)
        node[tuple(cu.T)] = np.arange(len(cu))
        rows, cols, w = [], [], []
        for off in _OFFSETS:
            cv = cu + off
            keep = ((cv >= 0) & (cv < np.array(sub.shape))).all(1)
            u, v = cu[keep], cv[keep]
            nv = node[tuple(v.T)]
            keep = nv >= 0
            for t in range(1, s):                    # the fine cells in between
                mid = u[keep] * s + off * t
                keep[keep] = clear[tuple(mid.T)]
            rows.append(node[tuple(u[keep].T)])
            cols.append(nv[keep])
            w.append(np.full(keep.sum(), s * self.res * np.linalg.norm(off)))
        n = len(cu)
        rows, cols, w = (np.concatenate(a) if a else np.empty(0) for a in (rows, cols, w))
        csr = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
        xyz = self.centres(cu * s)
        self._graph = dict(csr=csr, xyz=xyz, tree=cKDTree(xyz) if n else None, n=n)
        return self._graph

    def attach(self, p, check_segment=True, skip_start=0.0):
        """Graph node to join p to: the nearest within attach_radius whose
        straight segment from p is valid. -1 if none."""
        g = self.graph()
        if g["tree"] is None:
            return -1
        near = g["tree"].query_ball_point(p, self.attach_radius)
        if not near:
            return -1
        near = np.asarray(near)
        near = near[np.argsort(np.linalg.norm(g["xyz"][near] - p, axis=1))]
        for i in near:
            if not check_segment or self.segment_valid(p, g["xyz"][i], skip_start):
                return int(i)
        return -1

    def field(self, src):
        """Dijkstra from src (the robot, or a view). None if src cannot join
        the graph. The robot may stand where the newest map has just put a
        surface too close, so its first 0.25 m are not checked."""
        src = np.asarray(src, float)
        node = self.attach(src, check_segment=True, skip_start=0.25)
        if node < 0:
            return None
        dist, pred = dijkstra(self.graph()["csr"], directed=False, indices=node,
                              return_predecessors=True)
        return Field(src, node, dist, pred)

    def travel(self, field, pts):
        """Travel distance from the field's source to each point (inf where unreachable)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        out = np.full(len(pts), np.inf)
        if field is None:
            return out
        xyz = self.graph()["xyz"]
        for k, p in enumerate(pts):
            i = self.attach(p)
            if i >= 0 and np.isfinite(field.dist[i]):
                out[k] = field.dist[i] + np.linalg.norm(p - xyz[i])
        return out

    def path(self, field, goal):
        """Waypoints src -> goal along the graph, shortcut wherever the straight
        segment is valid. None if goal is unreachable."""
        goal = np.asarray(goal, float)
        if field is None:
            return None
        i = self.attach(goal)
        if i < 0 or not np.isfinite(field.dist[i]):
            return None
        xyz = self.graph()["xyz"]
        chain = [i]
        while chain[-1] != field.node:
            chain.append(int(field.pred[chain[-1]]))
        pts = [field.src] + [xyz[c] for c in reversed(chain)] + [goal]
        out, a = [pts[0]], 0
        while a < len(pts) - 1:
            for b in range(len(pts) - 1, a, -1):
                if b == a + 1 or self.segment_valid(pts[a], pts[b], 0.25 if a == 0 else 0.0):
                    break
            out.append(pts[b])
            a = b
        return out
