"""Boundary-score inference on an arbitrary point cloud.

    python infer.py --cloud example/00876-mv2HUxq3B53_t000.npy --out out/
    python infer.py --cloud my_scan.ply --out out/          # no labels needed

What the network needs, and why - all of it is in `prepare_patch` and
`predict`:

1.  Z-UP, METRES.  The model was trained in one gravity-aligned frame and is
    NOT rotation invariant about x/y: floors are horizontal and walls are
    vertical in everything it saw. Yaw about z is fine (it was augmented over
    the full circle). Convert a live cloud into this frame; do not canonicalise
    per chunk.

2.  VOXELISE AT 0.05 m.  Anything denser wastes points and shifts the local
    neighbourhood statistics the score is built from; anything sparser reads
    as holes. One point per 5 cm cell.

3.  CUT PATCHES.  The model runs on patches, not whole scenes: the k nearest
    neighbours of a seed point, at most 4096 of them, discarded beyond a 2.0 m
    radius. Seeds come from farthest-point sampling so the patches tile the
    cloud; `--overlap` sets how many times over.

4.  RECENTRE AND RESCALE EACH PATCH.  Subtract the seed, snapped to the voxel
    grid (`round(seed/V)*V`, not the raw seed - the snap keeps the integer grid
    consistent between patches), then divide by 2.0. Both `coord` and `feat`
    are these normalised coordinates; there is no colour input.

5.  DEDUPLICATE GRID CELLS.  `grid_coord` must hold unique integer cells or
    spconv silently produces garbage.

6.  RUN IN FP32.  spconv's bf16/fp16 path fails here with "can't find suitable
    algorithm for 0". Training used bf16; inference cannot.

Output, per point:  ||delta|| is the boundary score in [0, 1] and delta/||delta||
is the outward bearing. The score is NOT rescaled by the 2.0 - only the input
coordinates were. Points covered by several patches take the mean, and the
spread between patches is reported as a self-consistency check.
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))

VOXEL = 0.05        # m, the grid the model was trained at
MAX_POINTS = 4096   # most points in one patch
MAX_RADIUS = 2.0    # m, patch is truncated here
NORM_SCALE = 2.0    # coordinates are divided by this; the score is not


def load_cloud(path):
    """-> (N,3) float32 xyz, (N,3) uint8 rgb, (N,) float32 gt score or None."""
    if path.endswith(".npy"):                      # a stored trajectory
        rec = np.load(path, allow_pickle=True).item()
        gt = rec["score"].astype(np.float32) if "score" in rec else None
        return rec["coord"].astype(np.float32), rec["rgb"].astype(np.uint8), gt
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, np.float32)
    rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() \
        else np.full((len(xyz), 3), 160, np.uint8)
    return xyz, rgb, None


def voxelise(xyz, rgb, v=VOXEL):
    """One point per cell, keeping the first - positions are not snapped."""
    g = np.floor(xyz / v).astype(np.int64)
    _, keep = np.unique(g, axis=0, return_index=True)
    keep.sort()
    return xyz[keep], rgb[keep], keep


def farthest_point_seeds(xyz, m, rng):
    sel = np.empty(m, np.int64)
    sel[0] = rng.integers(len(xyz))
    d = np.linalg.norm(xyz - xyz[sel[0]], axis=1)
    for i in range(1, m):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(xyz - xyz[sel[i]], axis=1))
    return sel


def prepare_patch(xyz, rows, seed):
    """Steps 3-5 for one patch. -> (rows, normalised coords, grid cells)."""
    rows = rows[np.linalg.norm(xyz[rows] - seed, axis=1) <= MAX_RADIUS]
    if len(rows) < 64:
        return None
    local = (xyz[rows] - np.round(seed / VOXEL) * VOXEL).astype(np.float32)
    g = np.floor(local / VOXEL).astype(np.int64)
    g -= g.min(0)                                   # spconv wants non-negative
    _, keep = np.unique(g, axis=0, return_index=True)
    keep.sort()
    return rows[keep], local[keep] / NORM_SCALE, g[keep].astype(np.int32)


def build_model(weight, device="cuda"):
    sys.path.insert(0, HERE)
    from delta_net import DeltaNet
    model = DeltaNet().to(device).eval()
    ck = torch.load(weight, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    missing, unexpected = model.load_state_dict(
        {k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False)
    assert not missing, f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}"
    return model


def predict(model, xyz, overlap=3.0, batch=12, seed=0, device="cuda"):
    """-> per-point score, outward direction, patch count, inter-patch spread."""
    rng = np.random.default_rng(seed)
    m = max(1, int(np.ceil(len(xyz) / MAX_POINTS * overlap)))
    centres = farthest_point_seeds(xyz, m, rng)
    _, neigh = cKDTree(xyz).query(xyz[centres], k=min(MAX_POINTS, len(xyz)), workers=-1)

    s1 = np.zeros(len(xyz))
    s2 = np.zeros(len(xyz))
    vec = np.zeros((len(xyz), 3))
    cov = np.zeros(len(xyz), np.int32)

    for start in range(0, m, batch):
        packed, coords, grids, sizes = [], [], [], []
        for j, row in enumerate(np.atleast_2d(neigh[start:start + batch])):
            p = prepare_patch(xyz, row, xyz[centres[start + j]])
            if p is None:
                continue
            rows, local, g = p
            packed.append(rows); coords.append(local); grids.append(g); sizes.append(len(rows))
        if not coords:
            continue
        inp = dict(
            coord=torch.from_numpy(np.concatenate(coords)).float().to(device),
            feat=torch.from_numpy(np.concatenate(coords)).float().to(device),
            grid_coord=torch.from_numpy(np.concatenate(grids)).int().to(device),
            offset=torch.tensor(np.cumsum(sizes), dtype=torch.int32).to(device),
        )
        with torch.no_grad():
            delta = model(inp)["delta"].float().cpu().numpy()      # fp32, see (6)
        score = np.linalg.norm(delta, axis=1)
        pos = 0
        for rows in packed:
            sl = slice(pos, pos + len(rows)); pos += len(rows)
            np.add.at(s1, rows, score[sl])
            np.add.at(s2, rows, score[sl] ** 2)
            np.add.at(vec, rows, delta[sl])
            cov[rows] += 1

    seen = cov > 0
    out_score = np.zeros(len(xyz), np.float32)
    out_score[seen] = s1[seen] / cov[seen]
    out_dir = np.zeros((len(xyz), 3), np.float32)
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    out_dir[seen] = (vec[seen] / np.maximum(n[seen], 1e-9)).astype(np.float32)
    spread = np.zeros(len(xyz), np.float32)
    multi = cov > 1
    spread[multi] = np.sqrt(np.maximum(
        s2[multi] / cov[multi] - (s1[multi] / cov[multi]) ** 2, 0))
    return out_score, out_dir, cov, spread


def ramp(t):
    """blue (interior) -> cyan -> green -> yellow -> red (edge)"""
    t = np.clip(np.nan_to_num(t), 0, 1)
    stops = np.array([[0.20, 0.30, 0.80], [0.0, 0.75, 0.85], [0.35, 0.80, 0.30],
                      [0.95, 0.85, 0.15], [0.90, 0.12, 0.10]])
    x = t * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def save_ply(path, xyz, col):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(xyz, np.float64))
    p.colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(path, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", default=os.path.join(HERE, "example",
                                                    "00876-mv2HUxq3B53_t000.npy"))
    ap.add_argument("--weight", default=os.path.join(HERE, "..", "model_weights",
                                                 "delta_net.pth"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--overlap", type=float, default=3.0,
                    help="how many times over the patches cover the cloud")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="score above which a point is called a frontier; 0.25 "
                         "beats 0.30 on F1 because the model shaves the peaks")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-voxelise", action="store_true",
                    help="cloud is already on the 0.05 m grid")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    xyz, rgb, gt = load_cloud(args.cloud)
    print(f"loaded {len(xyz):,} points from {args.cloud}")
    if not args.no_voxelise:
        n0 = len(xyz)
        xyz, rgb, keep = voxelise(xyz, rgb)
        if gt is not None:
            gt = gt[keep]
        print(f"  voxelised at {VOXEL} m: {n0:,} -> {len(xyz):,}")
    print(f"  extent {np.round(xyz.max(0) - xyz.min(0), 2)} m  "
          f"(z is up: {np.ptp(xyz[:, 2]):.1f} m should be storey-scale)")

    model = build_model(args.weight, args.device)
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} parameters")

    score, direction, cov, spread = predict(
        model, xyz, args.overlap, args.batch, device=args.device)
    seen = cov > 0

    print(f"\ncoverage {100*seen.mean():.1f}% of points, {cov[seen].mean():.1f} patches each")
    print(f"score    mean {score[seen].mean():.4f}  p90 {np.percentile(score[seen], 90):.3f}  "
          f"max {score.max():.3f}")
    print(f"         above {args.threshold}: {100*(score[seen] > args.threshold).mean():.1f}% "
          f"of points ({int((score[seen] > args.threshold).sum()):,})")
    print(f"agreement between overlapping patches: {spread[cov > 1].mean():.4f} "
          f"(low = the patch split is not changing the answer)")
    if gt is not None:
        g, p = gt[seen], score[seen]
        hit, flag = g > args.threshold, p > args.threshold
        tp = (hit & flag).sum()
        print(f"\nagainst the stored label:")
        print(f"  MAE {np.abs(p - g).mean():.4f}   on edges {np.abs(p - g)[hit].mean():.4f}"
              f"   corr {np.corrcoef(p, g)[0, 1]:+.3f}")
        print(f"  at {args.threshold}: precision {tp/max(flag.sum(),1):.3f}  "
              f"recall {tp/max(hit.sum(),1):.3f}")

    stem = os.path.join(args.out, os.path.splitext(os.path.basename(args.cloud))[0])
    save_ply(stem + "_1_input.ply", xyz, rgb / 255.0)
    save_ply(stem + "_2_score.ply", xyz, ramp(score))
    flagged = np.tile([0.42, 0.43, 0.45], (len(xyz), 1))
    flagged[score > args.threshold] = [0.90, 0.12, 0.10]
    save_ply(stem + "_3_flagged.ply", xyz, flagged)
    if gt is not None:
        save_ply(stem + "_4_label.ply", xyz, ramp(gt))
    np.savez_compressed(stem + "_pred.npz", points=xyz, rgb=rgb, score=score,
                        direction=direction, coverage=cov, spread=spread)
    print(f"\nwrote {stem}_*.ply and {stem}_pred.npz")


if __name__ == "__main__":
    main()
