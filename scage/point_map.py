"""The Delta-Net map: every observed surface point, one per 0.05 m cell.

This is kept apart from the navigation map on purpose. Wavemap answers "is
this space free" at 0.1 m. Delta-Net needs the surface itself, as points, on
the 0.05 m grid it was trained at (README step 2) - a denser cloud shifts the
neighbourhood statistics the score is built from, a sparser one reads as holes.

Cells are filled once: the first point to land in a cell keeps it, the same
rule `infer.voxelise` applies, so positions are never snapped or averaged.

Each point also keeps the vertical component of its surface normal, taken
from the depth image it came from (neighbouring pixels, oriented toward the
camera). Ceilings - horizontal, facing down - are what tells the flight space
(`scage.space`) inside the building from outside it.
"""

import numpy as np

_BITS = 21                  # per axis; 2^21 cells of 5 cm is +-52 km
_OFF = 1 << (_BITS - 1)


def cell_keys(xyz, voxel):
    """One int64 per point naming its grid cell."""
    g = np.floor(np.asarray(xyz) / voxel).astype(np.int64) + _OFF
    return (g[:, 0] << (2 * _BITS)) | (g[:, 1] << _BITS) | g[:, 2]


class PointMap:
    """Append-only surface cloud on a fixed voxel grid.

    Per point it keeps which frame first saw it (`eye`, an index into
    `eyes`, the camera centres) - the viewpoint generator orients surface
    normals toward that - plus the cached Delta-Net output and a `dirty` flag
    for points added since the last scoring.
    """

    def __init__(self, K, voxel=0.05, min_range=0.1, max_range=8.0):
        self.K = np.asarray(K, dtype=float)
        self.voxel = float(voxel)
        self.min_range = float(min_range)
        self.max_range = float(max_range)

        self.xyz = np.empty((0, 3), np.float32)
        self.rgb = np.empty((0, 3), np.uint8)
        self.eye = np.empty(0, np.int32)
        self.eyes = np.empty((0, 3), np.float64)
        self.score = np.empty(0, np.float32)
        self.direction = np.empty((0, 3), np.float32)
        self.dirty = np.empty(0, bool)
        self.covered = np.empty(0, bool)        # reached by at least one patch
        self.nz = np.empty(0, np.float32)       # world z of the unit normal, toward the
                                                # camera; NaN where the image gave none
        self.looked = np.empty(0, np.int16)     # frames that saw it and past it while
                                                # it was anomalous (ScagePipeline.detect)

        # cell lookup: keys kept sorted, with the row each one lives at
        self._keys = np.empty(0, np.int64)
        self._rows = np.empty(0, np.int64)

    def __len__(self):
        return len(self.xyz)

    def backproject(self, depth, W_T_C, rgb=None):
        """Depth image -> world points, keeping depths in (min_range, max_range].
        -> xyz (N,3), rgb (N,3), nz (N,): pixels in row-major order, with the
        robot camera's pinhole model (scage.viewer.create_camera)."""
        depth = np.asarray(depth, np.float32)
        H, W = depth.shape
        ok = (depth > self.min_range) & (depth <= self.max_range)
        K = self.K
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        P = np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1],
                      depth], -1)
        n_cam = self._normals(P, depth, ok)
        T = np.asarray(W_T_C, float)
        R, t = T[:3, :3].astype(np.float32), T[:3, 3].astype(np.float32)
        xyz = P[ok] @ R.T + t
        nz = n_cam[ok] @ R[2]                # world z of each normal
        if rgb is None:
            col = np.full((len(xyz), 3), 160, np.uint8)
        else:
            col = np.asarray(rgb)[ok].astype(np.uint8)
        return xyz.astype(np.float32), col, nz.astype(np.float32)

    @staticmethod
    def _normals(P, depth, ok, max_jump=0.1):
        """Camera-frame unit normals from central differences, turned toward
        the camera. NaN at the border, next to invalid pixels, and across depth
        jumps of more than max_jump (relative) - those straddle two surfaces."""
        n = np.full(P.shape, np.nan, np.float32)
        c = (slice(1, -1), slice(1, -1))
        du = P[1:-1, 2:] - P[1:-1, :-2]
        dv = P[2:, 1:-1] - P[:-2, 1:-1]
        nc = np.cross(du, dv)
        z = depth[c]
        good = ok[c] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1]
        for nb in (depth[1:-1, 2:], depth[1:-1, :-2], depth[2:, 1:-1], depth[:-2, 1:-1]):
            good &= np.abs(nb - z) <= max_jump * z
        norm = np.linalg.norm(nc, axis=-1)
        good &= norm > 1e-12
        nc = nc / np.where(norm > 0, norm, 1.0)[..., None]
        flip = (nc * P[c]).sum(-1) > 0               # facing away from the camera
        nc[flip] *= -1
        nc[~good] = np.nan
        n[c] = nc
        return n

    def insert_depth(self, depth, W_T_C, rgb=None):
        xyz, col, nz = self.backproject(depth, W_T_C, rgb)
        return self.insert_points(xyz, np.asarray(W_T_C)[:3, 3], col, nz)

    def insert_points(self, xyz, eye, rgb=None, nz=None):
        """Add the points whose cells are still empty. -> number added."""
        self.eyes = np.vstack([self.eyes, np.asarray(eye, float).reshape(1, 3)])
        if len(xyz) == 0:
            return 0
        xyz = np.asarray(xyz, np.float32)
        if rgb is None:
            rgb = np.full((len(xyz), 3), 160, np.uint8)
        if nz is None:
            nz = np.full(len(xyz), np.nan, np.float32)

        keys, first = np.unique(cell_keys(xyz, self.voxel), return_index=True)
        pos = np.searchsorted(self._keys, keys)
        seen = pos < len(self._keys)
        seen[seen] = self._keys[pos[seen]] == keys[seen]

        idx = np.flatnonzero(~seen)
        idx = idx[np.argsort(first[idx])]           # keep the frame's pixel order
        n0, n = len(self.xyz), len(idx)
        if n == 0:
            return 0
        pick = first[idx]

        self.xyz = np.concatenate([self.xyz, xyz[pick]])
        self.rgb = np.concatenate([self.rgb, np.asarray(rgb, np.uint8)[pick]])
        self.eye = np.concatenate([self.eye, np.full(n, len(self.eyes) - 1, np.int32)])
        self.score = np.concatenate([self.score, np.zeros(n, np.float32)])
        self.direction = np.concatenate([self.direction, np.zeros((n, 3), np.float32)])
        self.dirty = np.concatenate([self.dirty, np.ones(n, bool)])
        self.covered = np.concatenate([self.covered, np.zeros(n, bool)])
        self.nz = np.concatenate([self.nz, np.asarray(nz, np.float32)[pick]])
        self.looked = np.concatenate([self.looked, np.zeros(n, np.int16)])

        keys_all = np.concatenate([self._keys, keys[idx]])
        rows_all = np.concatenate([self._rows, np.arange(n0, n0 + n)])
        order = np.argsort(keys_all, kind="stable")
        self._keys, self._rows = keys_all[order], rows_all[order]
        return n


class DualMapper:
    """Both maps (and the flight space's sky map) behind one insertion call.

    The app hands every captured frame - including the ones taken while
    following a path - to `mapper.insert_depth_to_buffer`. Wrapping the
    WaveMapper here puts each frame in the point map too. Everything else is
    forwarded to Wavemap.
    """

    def __init__(self, wave, points, space=None):
        self.wave = wave
        self.points = points
        self.space = space          # FlightSpace: no-return rays mark open sky
        self.frames = []            # W_T_C of every frame, for replay and evaluation

    def insert_depth_to_buffer(self, depth, transform, rgb=None):
        """rgb, the frame's colour image, only colours the point map's points
        (PointMap.rgb, for the saved cloud); which points it keeps does not change."""
        self.wave.insert_depth_to_buffer(depth=depth, transform=transform)
        self.points.insert_depth(depth, transform, rgb)
        if self.space is not None:
            self.space.observe_sky(depth, transform, self.points.K)
        self.frames.append(np.array(transform, float))

    def __getattr__(self, name):
        return getattr(self.wave, name)
