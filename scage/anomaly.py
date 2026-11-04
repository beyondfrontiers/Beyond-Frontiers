"""Delta-Net over the live map: per-point anomaly score and outward bearing.

Patch cutting, normalisation and the model are `infer.py`'s own
(`farthest_point_seeds`, `prepare_patch`, `build_model`), so the network sees
exactly what it was validated on. What is added here:

- `agg`: how a point covered by several patches combines them. `mean` is what
  infer.py and the README numbers use; `max` is what the paper describes.
- `seed_mask`: restrict patch seeds to part of the cloud, for re-scoring only
  around newly added points.
- `minmax`: the paper's Eq. 2 global normalisation, as an option.

Units: ||delta|| is the score, already in [0, 1]. Only the input coordinates
are divided by NORM_SCALE (inside prepare_patch); nothing is multiplied back
on the way out. The bearing is normalised explicitly - the raw vector carries
the score in its length.
"""

import os

import numpy as np
import torch
from scipy.spatial import cKDTree

import scage  # noqa: F401  (path setup)
from infer import MAX_POINTS, build_model, farthest_point_seeds, prepare_patch

WEIGHT = scage.WEIGHT


def minmax(score, valid=None):
    """Paper Eq. 2: rescale to [0, 1] over the valid points."""
    valid = np.ones(len(score), bool) if valid is None else valid
    out = np.zeros_like(score)
    if valid.any():
        lo, hi = score[valid].min(), score[valid].max()
        out[valid] = (score[valid] - lo) / max(hi - lo, 1e-9)
    return out


class AnomalyScorer:
    def __init__(self, weight=WEIGHT, device="cuda", overlap=3.0, batch=12,
                 agg="mean", seed=0):
        assert agg in ("mean", "max"), agg
        self.model = build_model(weight, device)
        self.device = device
        self.overlap = float(overlap)
        self.batch = int(batch)
        self.agg = agg
        self.rng = np.random.default_rng(seed)

    def score(self, xyz, seed_mask=None):
        """-> score (N,), unit bearing (N,3), patch count per point (N,).

        Points no patch reached get score 0, bearing 0 and count 0. With a
        seed_mask, patches are seeded only on those points, enough of them to
        tile that part `overlap` times over; they still gather neighbours from
        the whole cloud.
        """
        xyz = np.asarray(xyz, np.float32)
        n = len(xyz)
        score = np.zeros(n, np.float32)
        bearing = np.zeros((n, 3), np.float32)
        cov = np.zeros(n, np.int32)
        pool = np.arange(n) if seed_mask is None else np.flatnonzero(seed_mask)
        if n < 64 or len(pool) == 0:
            return score, bearing, cov

        m = max(1, int(np.ceil(len(pool) / MAX_POINTS * self.overlap)))
        m = min(m, len(pool))
        centres = pool[farthest_point_seeds(xyz[pool], m, self.rng)]
        _, neigh = cKDTree(xyz).query(xyz[centres], k=min(MAX_POINTS, n), workers=-1)
        neigh = np.atleast_2d(neigh)

        acc = np.zeros(n) if self.agg == "mean" else np.full(n, -np.inf)
        vec = np.zeros((n, 3))
        for start in range(0, m, self.batch):
            rows_list, coords, grids = [], [], []
            for j, row in enumerate(neigh[start:start + self.batch]):
                p = prepare_patch(xyz, row, xyz[centres[start + j]])
                if p is None:
                    continue
                rows, local, g = p
                rows_list.append(rows); coords.append(local); grids.append(g)
            if not coords:
                continue
            c = torch.from_numpy(np.concatenate(coords)).float().to(self.device)
            inp = dict(
                coord=c, feat=c,
                grid_coord=torch.from_numpy(np.concatenate(grids)).int().to(self.device),
                offset=torch.tensor(np.cumsum([len(r) for r in rows_list]),
                                    dtype=torch.int32).to(self.device),
            )
            with torch.no_grad():                       # fp32: spconv's half path fails
                delta = self.model(inp)["delta"].float().cpu().numpy()
            s = np.linalg.norm(delta, axis=1)
            pos = 0
            for rows in rows_list:
                sl = slice(pos, pos + len(rows)); pos += len(rows)
                if self.agg == "mean":
                    np.add.at(acc, rows, s[sl])
                else:
                    np.maximum.at(acc, rows, s[sl])
                np.add.at(vec, rows, delta[sl])
                np.add.at(cov, rows, 1)

        seen = cov > 0
        score[seen] = acc[seen] / cov[seen] if self.agg == "mean" else acc[seen]
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        bearing[seen] = vec[seen] / np.maximum(norm[seen], 1e-9)
        return score, bearing, cov
