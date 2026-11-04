"""Delta-Net on a single example: score a point cloud, cluster what it flags as
the edge of observed space, place one camera target per cluster and show them.
This is SCAGE's detection step on its own, without exploring - as FrontierNet's
single-image demo did for its frontiers.

    python demo_single.py                                    # a chunk of HM3D scene 00876
    python demo_single.py --depth examples/d455_desk.npz     # one RealSense D455 depth map
    python demo_single.py --depth examples/d455_office.npz --out output/demo --no_vis

For a depth map the score is first shown on the image itself (RGB | score,
any key to go on), as FrontierNet's demo showed its 2D result first. The 3D
window then shows the cloud coloured by score (blue -> red), a magenta sphere on
each anomalous cluster, the safe camera target looking at it (yellow -> red by
the anomaly mass it covers) and the capturing camera in blue. A cluster with no
safe target shows its Eq. 3 pose in grey; the terminal says why it was
rejected. Press C to swap the score colours for the camera's own.

Input
-----
1. A cloud (--cloud), already in the frame the network was trained in, as
   deltanet/example is: it goes in as it is. A .npy chunk carries the camera
   positions it was captured from ("eyes"); for a .ply give them with --eye.
2. A depth map (--depth, see examples/d455_*.npz). It is in the depth camera's
   optical frame: x right, y down, z forward. The network was trained in
   Matterport3D / HM3D's frame - z up along gravity, metres, any heading - and
   is not invariant to tilt, so the depth map is brought into that frame first:
     - flying pixels (mixed depths along object edges) are removed;
     - the rest is backprojected with the camera intrinsics, between
       MIN_RANGE and --max_range;
     - and rotated so that z is up - "up" is gravity in the camera frame,
       stored with the example - with the floor at z = 0 ("camera_height")
       and x along the camera's heading.
   A depth example holds depth (uint16, H x W) and depth_scale (m per unit),
   K (3x3 intrinsics), up (3,), camera_height, and rgb (H x W x 3, aligned to
   the depth, for display only - the network takes no colour).

Then, for both: one point per 0.05 m cell, Delta-Net (deltanet/infer.py),
threshold at tau, ball-query clusters (scage.clustering), and per cluster the
pose of paper Eq. 3 (scage.viewpoints): the camera r / tan(fov / 2) (at least
d_min) from the cluster's centre, along the surface normal, on the side it was
seen from, looking at it.

Without a navigation map, targets are checked against what this capture alone
saw (SeenSpace): the cloud's points are the obstacles, and space is known free
where a camera saw through it to a surface behind. One depth map sees little
free space - a wedge below the top of its view, short of the first surfaces -
so many clusters get no safe target from it; exploring is what adds the rest.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

import scage  # noqa: F401  (path setup)
from infer import VOXEL, build_model, load_cloud, predict, ramp, save_ply, voxelise
from scage import load_config
from scage.clustering import find_clusters
from scage.point_map import cell_keys
from scage.viewer import camera_vis_with_cylinders
from scage.viewpoints import ViewpointGenerator

MIN_RANGE = 0.3                                 # m, nearest depth kept
SIM_FOV = 2 * np.arctan(480 / (2 * 300.0))      # the simulated robot camera, 77.32 deg


# ---- depth map -> the network's frame ---------------------------------------

def filter_flying_pixels(depth_map, threshold=0.1, erosion=1):
    d = depth_map.astype(np.float32)
    ex = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    ey = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    mask = (ex > threshold) | (ey > threshold)
    if erosion > 0:
        k = np.ones((erosion * 2 + 1, erosion * 2 + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    out = d.copy()
    out[mask] = 0.0
    return out


def level_pose(up, height):
    """W_T_C of a camera whose gravity "up" is known in its own frame: world z
    along up, world x along the camera's forward axis levelled, and the camera
    `height` above z = 0."""
    z = np.asarray(up, float) / np.linalg.norm(up)
    x = np.array([0.0, 0.0, 1.0]) - z[2] * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z])    # rows: world axes in camera coordinates
    T[2, 3] = float(height)
    return T


def depth_to_cloud(rec, max_range):
    """A depth example -> xyz (N,3) in the network's frame, rgb (N,3), W_T_C,
    and the (H,W) mask of the pixels the points came from."""
    depth = filter_flying_pixels(rec["depth"].astype(np.float32) * float(rec["depth_scale"]))
    K = rec["K"]
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ok = (depth > MIN_RANGE) & (depth <= max_range)
    z = depth[ok]
    P = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], -1)
    W_T_C = level_pose(rec["up"], rec["camera_height"])
    xyz = P @ W_T_C[:3, :3].T + W_T_C[:3, 3]
    return xyz.astype(np.float32), rec["rgb"][ok], W_T_C, ok


def score_image(rgb, mask, xyz_px, xyz, score):
    """RGB | score: each pixel takes the score of the 0.05 m cell its point fell
    in (Delta-Net scores one point per cell). -> BGR image"""
    k = cell_keys(xyz, VOXEL)
    order = np.argsort(k)
    px = score[order[np.searchsorted(k[order], cell_keys(xyz_px, VOXEL))]]
    base = rgb.astype(np.float32) / 255.0
    over = base * 0.35
    over[mask] += 0.65 * ramp(px)
    return (np.hstack([base, over])[..., ::-1] * 255).astype(np.uint8)


# ---- free space from one capture --------------------------------------------

class SeenSpace:
    """What one capture says about space, with the methods ViewpointGenerator
    checks targets with (as scage.space.FlightSpace has them).

    Obstacles are the cloud's own points. A point is known free when a camera
    saw past it: along the ray from that camera through the point, within
    range, the first surface comes after the point. Where a ray found no
    surface (out of view, past the range, glass) nothing is known."""

    def __init__(self, xyz, eyes, bounds, max_range, min_dist2occ=0.2, step=0.05):
        self.tree = cKDTree(xyz)
        self.eyes = np.atleast_2d(np.asarray(eyes, float))
        self.bounds = np.asarray(bounds, float).reshape(3, 2)
        self.min_dist2occ = float(min_dist2occ)
        self.hit = float(step)              # a ray this close to a point has reached it
        self.ts = np.arange(step, float(max_range) + 1e-9, step)

    def is_free(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        free = np.zeros(len(P), bool)
        for e in self.eyes:
            v = P - e
            L = np.linalg.norm(v, axis=1)
            S = e + (v / np.maximum(L, 1e-9)[:, None])[:, None, :] * self.ts[None, :, None]
            d, _ = self.tree.query(S.reshape(-1, 3), distance_upper_bound=self.hit, workers=-1)
            near = (d <= self.hit).reshape(len(P), -1)
            first = np.where(near.any(1), self.ts[near.argmax(1)], np.inf)
            free |= np.isfinite(first) & (first > L + self.hit)
        return free

    def reason(self, p):
        """None if p is a valid camera position, else the first rule it breaks."""
        p = np.asarray(p, float)
        if not ((p >= self.bounds[:, 0]) & (p <= self.bounds[:, 1])).all():
            return "bounds"
        if not self.is_free(p)[0]:
            return "unknown"
        if self.tree.query(p)[0] <= self.min_dist2occ:
            return "clearance"
        return None


class AnySpace:
    """No checks but the flight band: where Eq. 3 alone puts each camera."""
    tree = None

    def __init__(self, z_lo, z_hi):
        self.bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [z_lo, z_hi]])

    def reason(self, p):
        return None

    def is_free(self, P):
        return np.ones(len(np.atleast_2d(P)), bool)


# ---- the demo -----------------------------------------------------------------

def load_input(args):
    """-> xyz, rgb, eyes (E,3), capture poses W_T_C (for drawing), fov, name, and
    for a depth map (image, pixel mask) for the score image."""
    if args.depth:
        rec = np.load(args.depth)
        xyz, rgb, W_T_C, mask = depth_to_cloud(rec, args.max_range)
        K, W = rec["K"], rec["depth"].shape[1]
        up = np.asarray(rec["up"], float)
        print(f"depth map {rec['depth'].shape[1]}x{rec['depth'].shape[0]}: {len(xyz):,} points "
              f"after the flying-pixel filter and {MIN_RANGE}-{args.max_range} m; camera "
              f"{float(rec['camera_height']):.2f} m above the floor, pitched "
              f"{np.degrees(np.arcsin(-up[2] / np.linalg.norm(up))):.0f} deg down")
        return xyz, rgb, W_T_C[None, :3, 3], [W_T_C], 2 * np.arctan(W / (2 * K[0, 0])), \
            os.path.splitext(os.path.basename(args.depth))[0], (rec["rgb"], mask)
    xyz, rgb, _ = load_cloud(args.cloud)
    if args.eye:
        eyes = np.array([[float(c) for c in e.split(",")] for e in args.eye])
    elif args.cloud.endswith(".npy"):
        eyes = np.asarray(np.load(args.cloud, allow_pickle=True).item()["eyes"], float)
    else:
        sys.exit("a .ply carries no camera positions: give them with --eye x,y,z")
    print(f"cloud {args.cloud}: {len(xyz):,} points from {len(eyes)} camera positions")
    return xyz, rgb, eyes, [], SIM_FOV, os.path.splitext(os.path.basename(args.cloud))[0], None


def place_targets(xyz, score, eyes, fov, cfg, max_range):
    """Clusters of anomalous points, a safe camera target for each that has one,
    and for the others the Eq. 3 pose with the check it fails."""
    sc = cfg["scage"]
    clusters = find_clusters(xyz, score, sc["tau"], sc["cluster_radius"], sc["min_cluster_pts"],
                             max_radius=sc["d_max"] * np.tan(fov / 2))
    lo, hi = np.minimum(xyz.min(0), eyes.min(0)), np.maximum(xyz.max(0), eyes.max(0))
    # the flight band of the config: heights above the floor, which is at z = 0
    bounds = [lo[0], hi[0], lo[1], hi[1], cfg["bounds"][4], cfg["bounds"][5]]
    kw = dict(d_min=sc["d_min"], normal_k=sc["normal_k"], max_vd_z=cfg["filter_max_vd_z"],
              target_clearance=sc["target_clearance"])
    gen = ViewpointGenerator(SeenSpace(xyz, eyes, bounds, max_range, cfg["min_dist2occ"]),
                             fov, **kw)
    # each point's camera: the nearest one (orients the surface normals)
    eye_ids = np.linalg.norm(xyz[:, None] - eyes[None], axis=2).argmin(1) if len(eyes) > 1 \
        else np.zeros(len(xyz), int)
    targets = gen.generate(xyz, score, eye_ids, eyes, clusters)
    safe = {id(t.rows) for t in targets}
    unchecked = ViewpointGenerator(AnySpace(bounds[4], bounds[5]), fov, **kw).generate(
        xyz, score, eye_ids, eyes, [c for c in clusters if id(c) not in safe])
    rejected = [(t, gen.check(t.position, t.anchor)) for t in unchecked]
    return clusters, targets, rejected, gen.last_report


def scene(xyz, rgb, score, targets, rejected, eyes, captures, fov, aspect):
    """-> the cloud, and everything else to draw with it."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(ramp(score))
    rest = []
    mass = np.array([score[t.rows].sum() * VOXEL ** 2 for t in targets])
    for t, m in zip(targets, mass):
        rest += camera_vis_with_cylinders(t.pose, wh_ratio=aspect, scale=0.4,
                                          fovx=np.degrees(fov), weight=m / mass.max(),
                                          radius=0.015)
    for t, _ in rejected:
        rest += camera_vis_with_cylinders(t.pose, wh_ratio=aspect, scale=0.3,
                                          fovx=np.degrees(fov), color=[0.55, 0.55, 0.55],
                                          radius=0.008)
    for t in targets + [t for t, _ in rejected]:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
        s.paint_uniform_color([1.0, 0.0, 0.6])
        s.compute_vertex_normals()
        rest.append(s.translate(t.anchor))
    for T in captures:
        rest += camera_vis_with_cylinders(T, wh_ratio=aspect, scale=0.4, fovx=np.degrees(fov),
                                          color=[0, 0, 1], radius=0.02)
    if not captures:
        for e in eyes:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            s.paint_uniform_color([0.0, 0.0, 1.0])
            rest.append(s.translate(e))
    return pc, rest


def view(xyz, eyes):
    """A look from behind and above the cameras, toward the cloud."""
    centre = xyz.mean(0)
    fwd = centre - eyes.mean(0)
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    front = -fwd + np.array([0.0, 0.0, 0.7])
    return dict(front=front / np.linalg.norm(front), lookat=centre, up=[0.0, 0.0, 1.0], zoom=0.45)


def show(pc, rest, rgb, score, cam):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Delta-Net: anomalous clusters and camera targets", 1280, 960)
    for g in [pc] + rest:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.set_front(cam["front"]); ctr.set_lookat(cam["lookat"])
    ctr.set_up(cam["up"]); ctr.set_zoom(cam["zoom"])
    colours = [ramp(score), rgb / 255.0]

    def swap(v):
        colours.reverse()
        pc.colors = o3d.utility.Vector3dVector(colours[0])
        v.update_geometry(pc)
        return True

    vis.register_key_callback(ord("C"), swap)
    vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cloud", default=os.path.join(scage.DELTANET, "example",
                                                     "00876-mv2HUxq3B53_t000.npy"),
                     help="a cloud in the network's frame (.npy chunk or .ply)")
    src.add_argument("--depth", help="a depth example (.npz, see examples/)")
    ap.add_argument("--eye", action="append",
                    help="x,y,z of a camera the cloud was captured from (repeatable)")
    ap.add_argument("--max_range", type=float, default=5.0,
                    help="depth beyond this is dropped (m); the D455 gets noisy past it")
    ap.add_argument("--config", default=scage.CONFIG)
    ap.add_argument("--out", help="also write <name>_score.ply, _targets.ply, _targets.json "
                                  "(and _score.jpg for a depth map) here")
    ap.add_argument("--no_vis", action="store_true", help="do not open the window")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = cfg["scage"]
    xyz_px, rgb, eyes, captures, fov, name, image = load_input(args)
    xyz, rgb, _ = voxelise(xyz_px, rgb)
    print(f"  voxelised at {VOXEL} m: {len(xyz_px):,} -> {len(xyz):,} points; z from "
          f"{xyz[:, 2].min():.2f} to {xyz[:, 2].max():.2f} m")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(scage.WEIGHT, device)
    # per point: score in [0, 1], unit bearing into the unobserved space (not
    # used here), and how many patches covered it
    score, bearing, cov, _ = predict(model, xyz, sc["overlap"], sc["batch"], device=device)
    flagged = score > sc["tau"]
    print(f"Delta-Net: score mean {score[cov > 0].mean():.3f}, {flagged.mean():.1%} of points "
          f"above tau = {sc['tau']}")
    stem = os.path.join(args.out or "", name)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if image is not None:
        img = score_image(*image, xyz_px, xyz, score)
        if args.out:
            cv2.imwrite(stem + "_score.jpg", img)
        if not args.no_vis:
            cv2.imshow("RGB | Delta-Net score (any key: 3D view)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    clusters, targets, rejected, rep = place_targets(xyz, score, eyes, fov, cfg, args.max_range)
    why = {}
    for _, r in rejected:
        why[r] = why.get(r, 0) + 1
    print(f"{len(clusters)} clusters -> {len(targets)} safe camera targets "
          f"({rep['primary_ok']} at the Eq. 3 pose, {len(targets) - rep['primary_ok']} at a "
          f"nearby safe one); no safe pose for {len(rejected)} (grey), their Eq. 3 pose: "
          f"{why or '-'}")
    for k, t in enumerate(sorted(targets, key=lambda t: -score[t.rows].sum())[:10]):
        print(f"  {k}: {t.n_pts:4d} pts, score {t.score:.2f}, camera at {t.position.round(2)} "
              f"{np.linalg.norm(t.position - t.anchor):.2f} m from the cluster")

    aspect = image[0].shape[1] / image[0].shape[0] if image is not None else 1.0
    pc, rest = scene(xyz, rgb, score, targets, rejected, eyes, captures, fov, aspect)
    if args.out:
        save_ply(stem + "_score.ply", xyz, ramp(score))
        mesh = o3d.geometry.TriangleMesh()
        for g in rest:
            mesh += g
        o3d.io.write_triangle_mesh(stem + "_targets.ply", mesh)
        with open(stem + "_targets.json", "w") as f:
            json.dump([dict(W_T_C=t.pose.round(5).tolist(), anchor=t.anchor.round(4).tolist(),
                            score=round(t.score, 4), points=t.n_pts,
                            mass=round(float(score[t.rows].sum() * VOXEL ** 2), 5),
                            safe=r is None, rejected_for=r)
                       for t, r in [(t, None) for t in targets] + rejected], f, indent=1)
        print(f"wrote {stem}_score.ply, _targets.ply, _targets.json"
              + (", _score.jpg" if image is not None else ""))
    if not args.no_vis:
        show(pc, rest, rgb, score, view(xyz, eyes))


if __name__ == "__main__":
    main()
