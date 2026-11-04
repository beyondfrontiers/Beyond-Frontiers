"""The Open3D windows the simulation runs in: cameras, depth capture, frustum
drawing and the manual-mode keys (WASD move, QZ height, JL turn, K print the
camera, C show RGB-D).

From FrontierNet (utils/vis_utils.py), unchanged; only the functions SCAGE
uses are kept.
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap

from scage.geometry import pose_difference, rot2quat


def create_camera(H, W, focal):
    fx, fy = focal, focal
    cx, cy = W / 2.0 - 0.5, H / 2.0 - 0.5
    return o3d.camera.PinholeCameraIntrinsic(
        width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy
    )


def set_viewpoint_ctr(vis, z_near=0.02, z_far=15.0):

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(z_far)
    ctr.set_constant_z_near(z_near)
    return ctr


def create_interactive_vis(
    H,
    W,
    camera_intrinsic,
    camera_extrinsic=np.eye(4),
    show_back_face=False,
    light_on=False,
    z_near=0.02,
    z_far=15.0,
):

    vis = o3d.visualization.VisualizerWithKeyCallback()
    assert H == camera_intrinsic.height
    assert W == camera_intrinsic.width
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = show_back_face
    vis.get_render_option().light_on = light_on
    ctr = set_viewpoint_ctr(vis, z_near, z_far)
    param = ctr.convert_to_pinhole_camera_parameters()
    param.intrinsic = camera_intrinsic
    param.extrinsic = camera_extrinsic
    success = ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    assert success
    return vis


def is_vis_moving(vis, previous_pose, trans_thre=0.02, rot_thre=0.05):
    state_v2 = get_vis_state(vis)
    cam_pose_v2 = np.linalg.inv(state_v2["cam_extrinsic"])
    trans_diff, rot_diff = pose_difference(
        previous_pose.reshape(1, 4, 4), cam_pose_v2.reshape(1, 4, 4)
    )

    if trans_diff[0, 0] > trans_thre or rot_diff[0, 0] > rot_thre:
        return True
    else:
        return False


def get_vis_state(vis):
    """Get the state of the visualizer
    returns: state dict:
    {
        "cam_intrinsic": PinholeCameraIntrinsic,
        "cam_extrinsic": 4x4 np.array,
    }
    """
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    cam_intrinsic = param.intrinsic
    cam_extrinsic = param.extrinsic
    return {"cam_intrinsic": cam_intrinsic, "cam_extrinsic": cam_extrinsic}


def set_vis_cam_ex(vis, cam_extrinsic):
    # get the original camera intrinsic
    intrinsic = get_vis_state(vis)["cam_intrinsic"]
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    param.extrinsic = cam_extrinsic
    param.intrinsic = intrinsic
    success = ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    assert success
    return


def set_vis_cam_intr(vis, cam_intrinsic):
    """Set the camera intrinsic of the visualizer
    vis: Open3D Visualizer
    cam_intrinsic: PinholeCameraIntrinsic
    """
    extrinsic = get_vis_state(vis)["cam_extrinsic"]
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    param.intrinsic = cam_intrinsic
    param.extrinsic = extrinsic
    success = ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    assert success
    return


## PERCEPTION ##
def capture_rgb(vis, return_rgb_type="np"):
    """Capture the RGB image from the visualizer
    vis: Open3D Visualizer
    return_rgb_type: str, default "np"
    returns: dict [np.array or o3d.geometry.Image, camera intrinsics, extrinsics]
    """
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic_matrix = cam_params.extrinsic.copy()
    intrinsic_matrix = cam_params.intrinsic.intrinsic_matrix

    color_image = vis.capture_screen_float_buffer(True)
    color_image_np = (np.asarray(color_image) * 255).astype(np.uint8)
    color_image = o3d.geometry.Image(color_image_np)

    return_rgb = color_image_np if return_rgb_type == "np" else color_image

    return {
        "image": return_rgb,
        "intrinsics": intrinsic_matrix,
        "extrinsics": extrinsic_matrix,
    }


def capture_depth(vis, return_depth_type="np"):
    """Capture the depth image from the visualizer
    vis: Open3D Visualizer
    return_depth_type: str, default "np"
    returns: dict [np.array or o3d.geometry.Image, camera intrinsics, extrinsics]
    """

    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic_matrix = cam_params.extrinsic.copy()
    intrinsic_matrix = cam_params.intrinsic.intrinsic_matrix

    depth_image = vis.capture_depth_float_buffer(True)
    depth_image_np = np.asarray(depth_image).astype(np.float32)
    depth_image = o3d.geometry.Image((depth_image_np * 255).astype(np.uint16))

    return_depth = depth_image_np if return_depth_type == "np" else depth_image

    return {
        "depth": return_depth,
        "intrinsics": intrinsic_matrix,
        "extrinsics": extrinsic_matrix,
    }


def capture_rgbd_and_visualize(vis):
    rgb = capture_rgb(vis, return_rgb_type="np")
    depth = capture_depth(vis, return_depth_type="np")
    # visualize the rgb and depth side by side, depth, use jet colormap
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb["image"])
    ax[0].axis("off")
    ax[0].set_title("RGB")
    ax[1].imshow(depth["depth"], cmap="jet")
    ax[1].axis("off")
    ax[1].set_title("Depth")
    plt.show()
    return False


def load_mesh(meshfile):
    mesh = o3d.io.read_triangle_mesh(meshfile, enable_post_processing=True)
    print("load mesh: {}".format(mesh))
    return mesh


def create_cylinder_between_points(
    point1, point2, radius=0.01, resolution=20, color=(0, 0, 0)
):
    """
    Create a cylinder between two 3D points.
    return: An Open3D TriangleMesh representing the cylinder.
    """
    # Compute the vector between the two points
    vec = point2 - point1
    length = np.linalg.norm(vec)
    if length == 0:
        return None

    # Create a cylinder oriented along the z-axis
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution
    )
    cylinder.paint_uniform_color(color)
    cylinder.compute_vertex_normals()

    # Compute rotation to align the cylinder with the vector
    z_axis = np.array([0, 0, 1])  # default cylinder direction
    axis = np.cross(z_axis, vec)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0:
        axis = axis / axis_len
        angle = np.arccos(np.dot(z_axis, vec) / length)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(R, center=(0, 0, 0))

    # Translate the cylinder to its correct position
    midpoint = (point1 + point2) / 2
    cylinder.translate(midpoint)

    return cylinder


def camera_vis_with_cylinders(
    transform: np.ndarray,
    wh_ratio: float = 4.0 / 3.0,
    scale: float = 1.0,
    fovx: float = 90.0,
    weight: float = 1.0,
    color: tuple = None,
    radius=0.01,
    return_mesh=False,
):
    """
    A camera frustum for visualization using cylinders with a yellow-to-red color gradient based on weight.
    transfomr: 4x4 numpy array, camera to world
    return: A list of Open3D geometries representing the camera frustum.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"Transform Matrix must be 4x4, but got {transform.shape}")

    # Compute frustum points
    pw = np.tan(np.deg2rad(fovx / 2.0)) * scale
    ph = pw / wh_ratio
    all_points = np.array(
        [
            [0.0, 0.0, 0.0],  # Frustum apex
            [pw, ph, scale],  # Top right
            [pw, -ph, scale],  # Bottom right
            [-pw, ph, scale],  # Top left
            [-pw, -ph, scale],  # Bottom left
        ]
    )

    # Define frustum edges by connecting points
    line_indices = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],  # Apex to corners
            [1, 2],
            [1, 3],
            [3, 4],
            [2, 4],  # Frustum base edges
        ]
    )

    # Custom yellow-to-red colormap
    yellow_red_cmap = LinearSegmentedColormap.from_list(
        "yellow_red", ["#F6FF55", "#FF0202"]
    )

    if color is not None:
        my_color = color

    else:
        if weight > 0 or weight == 0:
            color_value = min(weight, 1.0)
            my_color = yellow_red_cmap(color_value)[
                :3
            ]  # Get RGB from custom colormap for all cylinders
        else:
            # put blue color for weight = 0
            my_color = [0, 0, 1]

    # Create cylinders for each line segment in the frustum
    cylinders = []
    for start_idx, end_idx in line_indices:
        start_point = all_points[start_idx]
        end_point = all_points[end_idx]

        # Create the cylinder with the same color
        cylinder = create_cylinder_between_points(
            start_point, end_point, radius=radius, color=my_color
        )
        if cylinder is not None:
            cylinders.append(cylinder)

    # Apply the transformation to all cylinders
    for cylinder in cylinders:
        cylinder.transform(transform)

    if return_mesh:
        m = cylinders[0]
        for c in cylinders[1:]:
            m += c
        return m
    return cylinders


def print_camera_params(vis):
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    print("Camera intrinsics:")
    print(cam_params.intrinsic.intrinsic_matrix)
    print("Camera extrinsics:")
    print(cam_params.extrinsic)
    print("extrinsic quaternion")
    rot = cam_params.extrinsic.copy()[:3, :3]
    print(rot2quat(rot))
    return False


def move_forward(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0.1, 0, 0)
    return False


def move_backward(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(-0.1, 0, 0)
    return False


def move_up(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, 0.1)
    return False


def move_down(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, -0.1)
    return False


def move_left(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, -0.1, 0)
    return False


def move_right(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0.1, 0)
    return False


def rotate_left(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(-10, 0, 0)
    return False


def rotate_right(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(10, 0, 0)
    return False


def register_basic_callbacks(vis):
    vis.register_key_callback(ord("K"), print_camera_params)
    vis.register_key_callback(ord("W"), move_forward)
    vis.register_key_callback(ord("A"), move_left)
    vis.register_key_callback(ord("D"), move_right)
    vis.register_key_callback(ord("S"), move_backward)
    vis.register_key_callback(ord("Q"), move_up)
    vis.register_key_callback(ord("Z"), move_down)
    vis.register_key_callback(ord("L"), rotate_right)
    vis.register_key_callback(ord("J"), rotate_left)
    vis.register_key_callback(ord("C"), capture_rgbd_and_visualize)
    return vis

