"""Pose and camera geometry: alignment, pose differences, pose <-> (pos, quat),
interpolation, and which voxels a camera sees.

From FrontierNet (utils/geometry.py, utils/planner_utils.py,
utils/mapping_utils.py), unchanged; only the functions SCAGE uses are kept.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

EPS = 1e-8  # numerical tolerance


def rot2quat(rot):
    quat = R.from_matrix(rot).as_quat()  # xyzw
    return quat


def _safe_normalize(v, eps=EPS):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector encountered.")
    return v / n


def _rot_between(u, v, eps=EPS):
    """
    Small-angle-safe rotation that takes `u` onto `v`
    (both assumed unit).
    """
    dot = np.dot(u, v)
    if dot > 1 - eps:  # already aligned
        return R.identity()
    if dot < -1 + eps:  # opposite; pick any orthogonal axis
        # choose an axis least parallel to u for numerical stability
        helper = (
            np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )
        axis = _safe_normalize(np.cross(u, helper))
        return R.from_rotvec(axis * np.pi)

    axis = _safe_normalize(np.cross(u, v))
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis * angle)


def compute_alignment_transforms(
    origins,
    align_vec,
    align_axis,
    appr_vec,
    appr_axis,
    refine=True,
    eps=EPS,
    reortho=True,
):
    """
    A function to compute a list of 4x4 homogeneous transforms
    that align a set of origins with a specified direction and approach vector.
    The transforms are computed such that its align_axis aligns with the align_vec,
    and its approach axis try to align with the appr_vec as much as possible.
    Parameters:
    ----------
    origins : list[np.ndarray]
        List of 3D points (origins) where the transforms will be applied.
    align_vec : np.ndarray
        The target direction vector to align with.
    align_axis : np.ndarray
        The axis along which the align_vec should be aligned.
    appr_vec : np.ndarray
        The target approach vector to align with as much as possible.
    appr_axis : np.ndarray
        The axis along which the appr_vec should be aligned as much as possible.
    refine : bool, optional
        If True, applies an additional refinement step to align the approach vector.
        Default is True.
    eps : float, optional
        Numerical tolerance for vector normalization and rotation calculations.
        Default is 1e-8.
    reortho : bool, optional
        If True, re-orthonormalizes the resulting rotation matrix to ensure it is orthonormal.
        This can help eliminate numerical drift in the rotation matrix.
        Default is True.

    Returns
    -------
    list[np.ndarray] one 4x4 homogeneous transform per origin.
    """

    # align_axis and appr_axis should be orthonormal unit vectors.
    if not (
        np.isclose(np.linalg.norm(align_axis), 1.0, atol=eps)
        and np.isclose(np.linalg.norm(appr_axis), 1.0, atol=eps)
    ):
        raise ValueError("align_axis and appr_axis must be unit vectors.")
    if not (np.isclose(np.dot(align_axis, appr_axis), 0.0, atol=eps)):
        raise ValueError("align_axis and appr_axis must be orthogonal.")

    # upfront normalisation of target directions
    align_vec = _safe_normalize(align_vec, eps)
    appr_vec = _safe_normalize(appr_vec, eps)
    align_axis = _safe_normalize(align_axis, eps)
    appr_axis = _safe_normalize(appr_axis, eps)

    T_list = []

    # primary rotation: align align_axis → align_vec
    R1 = _rot_between(align_axis, align_vec, eps)

    for origin in origins:
        # ------------------------------------------------------------------
        # Optional refinement: twist about new align_vec so
        # (R1·appr_axis) lines up with appr_vec as much as possible.
        # ------------------------------------------------------------------
        if refine:
            new_appr_axis = R1.apply(appr_axis)

            # project both into the plane ⟂ align_vec
            proj_a = new_appr_axis - np.dot(new_appr_axis, align_vec) * align_vec
            proj_b = appr_vec - np.dot(appr_vec, align_vec) * align_vec

            n_a = np.linalg.norm(proj_a)
            n_b = np.linalg.norm(proj_b)

            if n_a > eps and n_b > eps:
                proj_a /= n_a
                proj_b /= n_b

                dot = np.clip(np.dot(proj_a, proj_b), -1.0, 1.0)
                angle = np.arccos(dot)

                # choose sign so the rotation has the shorter direction
                if np.dot(np.cross(proj_a, proj_b), align_vec) < 0:
                    angle = -angle

                R2 = R.from_rotvec(align_vec * angle)
                R_comb = R2 * R1
            else:
                # projections degenerate ⇒ skip refinement
                R_comb = R1
        else:
            R_comb = R1

        # optional re-orthonormalisation to kill tiny drift
        if reortho:
            U, _, Vt = np.linalg.svd(R_comb.as_matrix())
            R_comb = R.from_matrix(U @ Vt)

        # build homogeneous transform
        T = np.eye(4)
        T[:3, :3] = R_comb.as_matrix()
        T[:3, 3] = origin
        T_list.append(T)

    return T_list


def pose_difference(tf_1, tf_2):
    """
    Compute pairwise translation and rotation differences between two sets of transformation matrices.

    Args:
        tf_1: (N, 4, 4) numpy array
        tf_2: (M, 4, 4) numpy array

    Returns:
        translation_diff: (N, M) numpy array of L2 translation differences
        rotation_diff: (N, M) numpy array of rotation angle differences (in radians)
    """
    assert tf_1.shape[1:] == (4, 4) and tf_2.shape[1:] == (
        4,
        4,
    ), "Input tensors must be of shape (N, 4, 4) and (M, 4, 4)"
    N, M = tf_1.shape[0], tf_2.shape[0]

    # Extract translations
    trans_1 = tf_1[:, :3, 3]  # (N, 3)
    trans_2 = tf_2[:, :3, 3]  # (M, 3)

    # Compute translation differences (N, M)
    trans_diff = np.linalg.norm(trans_1[:, None, :] - trans_2[None, :, :], axis=-1)

    # Extract rotations
    rot_1 = tf_1[:, :3, :3]  # (N, 3, 3)
    rot_2 = tf_2[:, :3, :3]  # (M, 3, 3)

    # Compute relative rotation matrices: R_rel = R2 @ R1.T for all pairs
    rot_1_T = rot_1.transpose(0, 2, 1)  # (N, 3, 3)
    rel_rot = np.einsum("mij,njk->nmik", rot_2, rot_1_T)

    # Flatten to compute rotation differences using scipy
    rel_rot_flat = rel_rot.reshape(-1, 3, 3)
    relative_rotations = R.from_matrix(rel_rot_flat)
    rot_diff = np.linalg.norm(relative_rotations.as_rotvec(), axis=1).reshape(N, M)

    return trans_diff, rot_diff


def pose2posquat(pose):
    """
    Convert a 4x4 pose matrix to position and quaternion.

    Args:
        pose (np.ndarray): A 4x4 numpy array representing the pose.

    Returns:
        tuple: A dictionary containing position (np.ndarray) and quaternion (np.ndarray).
    """
    if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
        raise ValueError("Input pose must be a 4x4 numpy array.")

    position = pose[:3, 3]
    rotation_matrix = pose[:3, :3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    return {"pos": position, "quat": quaternion}


def posquat2pose(posquat):
    """
    Convert position and quaternion to a 4x4 pose matrix.

    Args:
        pos (np.ndarray): A 3-element numpy array representing the position.
        quat (np.ndarray): A 4-element numpy array representing the quaternion.

    Returns:
        np.ndarray: A 4x4 numpy array representing the pose.
    """
    pos = np.array(posquat["pos"])
    quat = np.array(posquat["quat"])
    if not isinstance(pos, np.ndarray) or pos.shape != (3,):
        raise ValueError("Position must be a 3-element numpy array.")
    if not isinstance(quat, np.ndarray) or quat.shape != (4,):
        raise ValueError("Quaternion must be a 4-element numpy array.")

    rotation_matrix = R.from_quat(quat).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = pos

    return pose


def slerp(quat0, quat1, t):
    # SLERP (Spherical Linear Interpolation) for quaternions
    dot = np.dot(quat0, quat1)

    if dot < 0.0:
        quat1 = -quat1
        dot = -dot

    if dot > 0.9995:
        result = quat0 + t * (quat1 - quat0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    quat2 = quat1 - quat0 * dot
    quat2 = quat2 / np.linalg.norm(quat2)

    return quat0 * np.cos(theta) + quat2 * np.sin(theta)


def interpolate_waypoints(start, end, num_interp_points=20):
    """
    Interpolate between start and end poses using linear position interpolation
    and SLERP for quaternion rotation.

    Parameters
    ----------
    start : dict
        {"pos": [x, y, z], "quat": [x, y, z, w]}
    end : dict
        {"pos": [x, y, z], "quat": [x, y, z, w]}
    num_interp_points : int
        Number of intermediate points (excluding start and end).

    Returns
    -------
    list of dict
        [start, ...intermediates..., end]
    """
    start_pos = np.asarray(start["pos"], dtype=float)
    end_pos = np.asarray(end["pos"], dtype=float)

    start_quat = np.asarray(start["quat"], dtype=float)
    end_quat = np.asarray(end["quat"], dtype=float)

    # Time steps (excluding endpoints for mids)
    t_values = np.linspace(0.0, 1.0, num_interp_points + 2)[1:-1]

    # Build path
    path = [{"pos": start_pos, "quat": start_quat}]
    for t in t_values:
        interp_pos = (1 - t) * start_pos + t * end_pos
        interp_quat = slerp(start_quat, end_quat, t)
        path.append({"pos": interp_pos, "quat": interp_quat})
    path.append({"pos": end_pos, "quat": end_quat})

    return path


def select_visible_points(points, K, T, img_size, max_depth=np.inf, min_depth_map=None):
    """
    Project 3D points to 2D image coordinates using camera intrinsics and extrinsics,
    returning only those that are visible and optionally closer than a depth map.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 3)), np.array([])

    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
    valid_points = np.ones(points.shape[0], dtype=bool)

    # Transform to camera coordinates
    points_cam = (T @ points_h.T).T[:, :3]  # (N, 3)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # Filter by depth range
    valid_points &= (z > 0) & (z <= max_depth)

    # Project to image plane
    u = np.round(K[0, 0] * x / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * y / z + K[1, 2]).astype(int)

    # Filter out-of-bounds pixels
    in_bounds = (u >= 0) & (u < img_size[1]) & (v >= 0) & (v < img_size[0])
    valid_points &= in_bounds

    # Filter based on depth map
    if min_depth_map is not None:
        depth_map_values = np.full(points.shape[0], np.nan)
        depth_map_values[in_bounds] = min_depth_map[v[in_bounds], u[in_bounds]]
        visible_mask = np.isnan(depth_map_values) | (z <= depth_map_values)
        valid_points &= visible_mask

    valid_indices = np.where(valid_points)[0]
    return points[valid_indices], valid_indices


def compute_visible_voxels(
    K, T, img_size, voxel_size=0.1, max_depth=np.inf, min_depth_map=None
):
    """
    Calculate the maximum number of voxels that can be visible in an image based on camera parameters and optional depth map.
    """

    # create voxel grid in camera coordinates
    z = np.arange(voxel_size, max_depth, voxel_size)
    x_range = int(0.8 * (max_depth * (img_size[1] / 2)) / K[0, 0])
    y_range = int(0.8 * (max_depth * (img_size[0] / 2)) / K[1, 1])
    x = np.arange(-x_range, x_range + voxel_size, voxel_size)
    y = np.arange(-y_range, y_range + voxel_size, voxel_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    voxel_centers = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  # (N, 3)

    # transform voxel centers to world coordinates
    voxel_centers_h = np.hstack((voxel_centers, np.ones((voxel_centers.shape[0], 1))))
    voxel_centers_world = (np.linalg.inv(T) @ voxel_centers_h.T).T[:, :3]

    # handle strict zero depth
    if min_depth_map is not None:
        min_depth_map = np.array(min_depth_map, dtype=np.float32)
        min_depth_map[abs(min_depth_map) < 1e-3] = max_depth

    # project to image and get visible ones
    valid_voxels, valid_indices = select_visible_points(
        voxel_centers_world,
        K=K,
        T=T,
        img_size=img_size,
        max_depth=max_depth,
        min_depth_map=min_depth_map,
    )

    return voxel_centers_world[valid_indices], valid_voxels.shape[0]
