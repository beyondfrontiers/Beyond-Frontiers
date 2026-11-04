"""Averaging camera orientations on SO(3).

Two nearby target views are clubbed into one by averaging their poses. Position
averages in R^3; orientation does not - rotations are not a vector space:

  - Euler angles wrap: yaw +170 deg and -170 deg average to 0 deg, the
    opposite direction, instead of 180 deg;
  - the element-wise mean of rotation matrices is not a rotation (it shrinks
    toward the origin as the views spread, det < 1).

Two proper means are used here (Hartley et al., "Rotation Averaging", IJCV 2013):

  chordal L2   argmin sum w_i ||R - R_i||_F^2: project sum w_i R_i back onto
               SO(3) with an SVD. Closed form; the same as Markley's quaternion
               average up to the metric.
  geodesic L2  (Karcher / Frechet mean) argmin sum w_i angle(R, R_i)^2, by
               Gauss-Newton on the tangent space, R <- R exp(mean_i log(R^T R_i)),
               started from the chordal mean.

For views within a few tens of degrees of each other the two agree to a small
fraction of a degree (tests/test_so3.py measures it). The mean of level
cameras (no roll) is in general slightly rolled: the level cameras are not a
subgroup of SO(3). `level` removes that, keeping the optical axis.
"""

import numpy as np


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp(w):
    """Rotation vector -> rotation matrix (Rodrigues)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def log(R):
    """Rotation matrix -> rotation vector, angle in [0, pi]."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * v
    if np.pi - th < 1e-4:
        # near pi the antisymmetric part vanishes; take the axis from R + I
        B = (R + np.eye(3)) / 2
        axis = B[np.argmax(np.diag(B))]
        axis = axis / np.linalg.norm(axis)
        if axis @ v < 0:
            axis = -axis
        return th * axis
    return th / (2 * np.sin(th)) * v


def angle(R1, R2):
    """Geodesic distance, radians."""
    return float(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)))


def chordal_mean(Rs, w=None):
    Rs = np.asarray(Rs, float)
    w = np.ones(len(Rs)) if w is None else np.asarray(w, float)
    M = np.einsum("i,ijk->jk", w, Rs)
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def karcher_mean(Rs, w=None, iters=50, tol=1e-12):
    """Weighted geodesic L2 mean. -> (R, iterations used)."""
    Rs = np.asarray(Rs, float)
    w = np.ones(len(Rs)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    R = chordal_mean(Rs, w)
    for it in range(1, iters + 1):
        step = sum(wi * log(R.T @ Ri) for wi, Ri in zip(w, Rs))
        R = R @ exp(step)
        if np.linalg.norm(step) < tol:
            return R, it
    return R, iters


def roll(R):
    """Roll of a CV camera (z forward, y down) about its optical axis, radians:
    the angle its x axis makes with the horizontal plane."""
    x, z = R[:, 0], R[:, 2]
    level_x = np.cross([0.0, 0.0, -1.0], z)         # x of the level camera with this axis
    n = np.linalg.norm(level_x)
    if n < 1e-9:                                     # looking straight up or down
        return 0.0
    level_x /= n
    return float(np.arctan2(np.cross(level_x, x) @ z, level_x @ x))


def level(R):
    """The level camera (no roll) with the same optical axis."""
    z = R[:, 2] / np.linalg.norm(R[:, 2])
    x = np.cross([0.0, 0.0, -1.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def mean_pose(poses, w=None):
    """Weighted mean of W_T_C camera poses: positions in R^3, orientations by
    the Karcher mean, then levelled. -> 4x4."""
    poses = np.asarray(poses, float)
    w = np.ones(len(poses)) if w is None else np.asarray(w, float)
    T = np.eye(4)
    T[:3, 3] = np.average(poses[:, :3, 3], axis=0, weights=w)
    T[:3, :3] = level(karcher_mean(poses[:, :3, :3], w)[0])
    return T
