"""Per-update snapshots, and the run's result, as .ply files any point-cloud
viewer opens.

    step_XXXX_score.ply     the Delta-Net map coloured by score (blue -> red)
    step_XXXX_clusters.ply  anomalous clusters in random colours, the rest grey
    step_XXXX_targets.ply   view frustums (yellow -> red by utility, the tour green),
                            a sphere at each view's anomaly centroid, the committed
                            goal and the robot in blue

At the end of a run (write_final):

    STEM_cloud.ply          the Delta-Net map in the robot camera's colours
    STEM_path.ply           the robot's path, as tubes from blue (start) to red (end)
"""

import os

import numpy as np
import open3d as o3d

from infer import ramp, save_ply
from scage.viewer import camera_vis_with_cylinders, create_cylinder_between_points


def _sphere(center, radius, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.paint_uniform_color(color)
    s.translate(np.asarray(center, float))
    return s


def write_update(out_dir, step, pipeline, robot_pose, fov_deg, goal_pose=None):
    stem = os.path.join(out_dir, f"step_{step:04d}")
    pm, last = pipeline.points, pipeline.last
    save_ply(stem + "_score.ply", pm.xyz, ramp(last["score"]))

    col = np.tile([0.55, 0.55, 0.55], (len(pm), 1))
    rng = np.random.default_rng(0)
    for rows in last["clusters"]:
        col[rows] = rng.uniform(0.15, 1.0, 3)
    save_ply(stem + "_clusters.ply", pm.xyz, col)

    m = pipeline.manager
    fts = m.valid_frontiers
    mesh = o3d.geometry.TriangleMesh()
    umax = max([ft.utility or 0.0 for ft in fts] + [1e-9])
    for ft in fts:
        mesh += camera_vis_with_cylinders(
            ft.pose6d, wh_ratio=1.0, scale=0.4, fovx=fov_deg,
            weight=(ft.utility or 0.0) / umax,
            color=[0.1, 0.8, 0.1] if ft.id in m.tour else None,
            radius=0.015, return_mesh=True)
        mesh += _sphere(ft.features["anchor"], 0.05, [1.0, 0.0, 0.6])
    if goal_pose is not None:
        mesh += camera_vis_with_cylinders(goal_pose, wh_ratio=1.0, scale=0.5, fovx=fov_deg,
                                          color=[0, 0.3, 1], radius=0.025, return_mesh=True)
    mesh += camera_vis_with_cylinders(robot_pose, wh_ratio=1.0, scale=0.5, fovx=fov_deg,
                                      color=[0, 0, 1], radius=0.02, return_mesh=True)
    o3d.io.write_triangle_mesh(stem + "_targets.ply", mesh)


def write_final(stem, points, frames, radius=0.03):
    """The Delta-Net map (PointMap) in colour, and the path through every
    captured camera pose (frames, W_T_C). -> the files written"""
    out = [stem + "_cloud.ply"]
    save_ply(out[0], points.xyz, points.rgb / 255.0)

    P = np.stack(frames)[:, :3, 3]
    n = max(len(P) - 2, 1)
    path = o3d.geometry.TriangleMesh()
    for i in range(len(P) - 1):
        if np.linalg.norm(P[i + 1] - P[i]) > 1e-3:          # not the turns on the spot
            t = i / n
            path += create_cylinder_between_points(
                P[i], P[i + 1], radius=radius, resolution=8,
                color=(1 - t) * np.array([0.0, 0.4, 1.0]) + t * np.array([1.0, 0.0, 0.0]))
    if len(path.vertices):
        out.append(stem + "_path.ply")
        o3d.io.write_triangle_mesh(out[1], path)
    return out
