import os
import argparse
import torch
import math
import numpy as np

from utils.print_fn import log
from utils import io_util, rend_util
from utils.checkpoints import sorted_ckpts
from models.frameworks import get_model

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp

def normalize(vec, axis=-1):
    return vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + 1e-9)

def view_matrix(
    forward: np.ndarray,
    up: np.ndarray,
    cam_location: np.ndarray):
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    mat = np.stack((rot_x, rot_y, rot_z, cam_location), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat

def look_at(
    cam_location: np.ndarray,
    point: np.ndarray,
    up=np.array([0., -1., 0.])          # openCV convention
    # up=np.array([0., 1., 0.])         # openGL convention
    ):
    # Cam points in positive z direction
    forward = normalize(point - cam_location)     # openCV convention
    # forward = normalize(cam_location - point)   # openGL convention
    return view_matrix(forward, up, cam_location)

def smoothed_motion_interpolation(full_range, num_samples, uniform_proportion=1/3.):
    half_acc_proportion = (1-uniform_proportion) / 2.
    num_uniform_acc = max(math.ceil(num_samples*half_acc_proportion), 2)
    num_uniform = max(math.ceil(num_samples*uniform_proportion), 2)
    num_samples = num_uniform_acc * 2 + num_uniform
    seg_velocity = np.arange(num_uniform_acc)
    seg_angle = np.cumsum(seg_velocity)
    # NOTE: full angle = 2*k*x_max + k*v_max*num_uniform
    ratio = full_range / (2.0*seg_angle.max()+seg_velocity.max()*num_uniform)
    # uniform acceleration sequence
    seg_acc = seg_angle * ratio

    acc_angle = seg_acc.max()
    # uniform sequence
    seg_uniform = np.linspace(acc_angle, full_range-acc_angle, num_uniform+2)[1:-1]
    # full sequence
    all_samples = np.concatenate([seg_acc, seg_uniform, full_range-np.flip(seg_acc)])
    return all_samples

def c2w_track_spiral(c2ws, num_views=10):
    #-----------------
    # Spiral path
    #   original nerf-like spiral path
    #-----------------

    center = c2ws[:, :3, 3].mean(0)
    forward = c2ws[:, :3, 2].sum(0)
    up = c2ws[:, :3, 1].sum(0)
    c2w_center = view_matrix(forward, up, center)

    up = c2ws[:, :3, 1].sum(0)
    rads = np.percentile(np.abs(c2ws[:, :3, 3]), 30, 0)
    focus_distance = np.mean(np.linalg.norm(c2ws[:, :3, 3], axis=-1))
    return c2w_track_spiral(c2w_center, up, rads, focus_distance * 0.8, zrate=0.0, rots=1, N=num_views)

def c2w_track_spiral_internal(c2w, up_vec, rads, focus: float, zrate: float, rots: int, N: int, zdelta: float = 0.):
    # TODO: support zdelta
    """generate camera to world matrices of spiral track, looking at the same point [0,0,focus]

    Args:
        c2w ([4,4] or [3,4]):   camera to world matrix (of the spiral center, with average rotation and average translation)
        up_vec ([3,]):          vector pointing up
        rads ([3,]):            radius of x,y,z direction, of the spiral track
        # zdelta ([float]):       total delta z that is allowed to change
        focus (float):          a focus value (to be looked at) (in camera coordinates)
        zrate ([float]):        a factor multiplied to z's angle
        rots ([int]):           number of rounds to rotate
        N ([int]):              number of total views
    """

    c2w_tracks = []
    rads = np.array(list(rads) + [1.])

    # focus_in_cam = np.array([0, 0, -focus, 1.])   # openGL convention
    focus_in_cam = np.array([0, 0, focus, 1.])  # openCV convention
    focus_in_world = np.dot(c2w[:3, :4], focus_in_cam)

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        cam_location = np.dot(
            c2w[:3, :4],
            # np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads    # openGL convention
            np.array([np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.]) * rads  # openCV convention
        )
        c2w_i = look_at(cam_location, focus_in_world, up=up_vec)
        c2w_tracks.append(c2w_i)
    return c2w_tracks

def c2w_track_sphericalspiral(c2ws, view_ids=[1,11,15], **kwargs):
    assert len(
        view_ids) == 3, 'please select three views on a small circle, in counter-clockwise(CCW) order (from above)'
    up_angle = np.pi / 3.
    n_rots = 2.2

    centers = c2ws[view_ids, :3, 3]
    centers_norm = np.linalg.norm(centers, axis=-1)
    radius = np.max(centers_norm)
    centers = centers * radius / centers_norm
    vec0 = centers[1] - centers[0]
    vec1 = centers[2] - centers[0]
    # the axis vertical to the small circle's area
    up_vec = normalize(np.cross(vec0, vec1))

    # key rotations of a spherical spiral path
    sphere_thetas = np.linspace(0, np.pi * 2. * n_rots, args.num_views)
    sphere_phis = np.linspace(0, up_angle, args.num_views)

    # use the origin as the focus center
    focus_center = np.zeros([3])

    # first rotate about up vec
    rots_theta = R.from_rotvec(sphere_thetas[:, None] * up_vec[None, :])
    render_centers = rots_theta.apply(centers[0])
    # then rotate about horizontal vec
    horizontal_vec = normalize(np.cross(render_centers - focus_center[None, :], up_vec[None, :], axis=-1))
    rots_phi = R.from_rotvec(sphere_phis[:, None] * horizontal_vec)
    render_centers = rots_phi.apply(render_centers)

    render_c2ws = look_at(render_centers, focus_center[None, :], up=-up_vec)
    debug = kwargs['debug'] if 'debug' in kwargs.keys() else False
    if debug:
        from vis_camera import visualize_cam_spherical_spiral
        intrinsics = kwargs['intrinsics']
        # plot camera path
        intr = intrinsics.data.cpu().numpy()
        extrs = np.linalg.inv(render_c2ws)
        visualize_cam_spherical_spiral(intr, extrs, up_vec, centers[0], focus_center, n_rots, up_angle)
    return render_c2ws

def c2w_track_smallcircle(c2ws, view_ids=[1,11,15], **kwargs):
    # ------------------
    # Small Circle Path:
    #   assume three input views are on a small circle, then interpolate along this small circle
    # ------------------
    assert len(view_ids) == 3, 'please select three views on a small circle, int CCW order (from above)'
    centers = c2ws[view_ids, :3, 3]
    centers_norm = np.linalg.norm(centers, axis=-1)
    radius = np.max(centers_norm)
    centers = centers * radius / centers_norm
    vec0 = centers[1] - centers[0]
    vec1 = centers[2] - centers[0]
    # the axis vertical to the small circle
    up_vec = normalize(np.cross(vec0, vec1))
    # length of the chord between c0 and c2
    len_chord = np.linalg.norm(vec1, axis=-1)
    # angle of the smaller arc between c0 and c1
    full_angle = np.arcsin(len_chord / 2 / radius) * 2.

    all_angles = smoothed_motion_interpolation(full_angle, args.num_views)

    rots = R.from_rotvec(all_angles[:, None] * up_vec[None, :])
    centers = rots.apply(centers[0])

    # get c2w matrices
    render_c2ws = look_at(centers, np.zeros_like(centers), up=-up_vec)
    debug = kwargs['debug'] if 'debug' in kwargs.keys() else False
    if debug:
        from vis_camera import visualize_cam_on_circle
        intrinsics = kwargs['intrinsics']
        # plot camera path
        intr = intrinsics.data.cpu().numpy()
        extrs = np.linalg.inv(render_c2ws)
        visualize_cam_on_circle(intr, extrs, up_vec, centers[0])

    return render_c2ws

def c2w_track_interpolation(c2ws, num_views, **kwargs):
    # -----------------
    # Interpolate path
    #   directly interpolate among all input views
    # -----------------
    # c2ws = c2ws[:25]  # NOTE: [:20] fox taxi dataset
    key_rots = R.from_matrix(c2ws[:, :3, :3])
    key_times = list(range(len(key_rots)))
    slerp = Slerp(key_times, key_rots)
    interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
    render_c2ws = []
    for i in range(num_views):
        time = float(i) / num_views * (len(c2ws) - 1)
        cam_location = interp(time)
        cam_rot = slerp(time).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_location
        render_c2ws.append(c2w)
    render_c2ws = np.stack(render_c2ws, axis=0)

    return render_c2ws

def c2w_track_greatcircle(c2ws, view01=[11,15], **kwargs):
    # ------------------
    # Great Circle Path:
    #   assume two input views are on a great circle, then interpolate along this great circle
    # ------------------
    # to interpolate along a great circle that pass through the c2w center of view0 and view1
    assert len(view01) == 2, 'please select two views on a great circle, in CCW order (from above)'
    view0, view1 = view01[0], view01[1]
    c0 = c2ws[view0, :3, 3]
    c0_norm = np.linalg.norm(c0)
    c1 = c2ws[view1, :3, 3]
    c1_norm = np.linalg.norm(c1)
    # the radius of the great circle
    # radius = (c0_norm+c1_norm)/2.
    radius = max(c0_norm, c1_norm)
    # re-normalize the c2w centers to be on the exact same great circle
    c0 = c0 * radius / c0_norm
    c1 = c1 * radius / c1_norm
    # the axis vertical to the great circle
    up_vec = normalize(np.cross(c0, c1))
    # length of the chord between c0 and c1
    len_chord = np.linalg.norm(c0 - c1, axis=-1)
    # angle of the smaller arc between c0 and c1
    full_angle = np.arcsin(len_chord / 2 / radius) * 2.

    all_angles = smoothed_motion_interpolation(full_angle, args.num_views)

    # get camera centers
    rots = R.from_rotvec(all_angles[:, None] * up_vec[None, :])
    centers = rots.apply(c0)

    # get c2w matrices
    render_c2ws = look_at(centers, np.zeros_like(centers), up=-up_vec)

    debug = kwargs['debug'] if 'debug' in kwargs.keys() else False
    if debug:
        from vis_camera import visualize_cam_on_circle
        intrinsics = kwargs['intrinsics']
        # plot camera path
        intr = intrinsics.data.cpu().numpy()
        extrs = np.linalg.inv(render_c2ws)
        visualize_cam_on_circle(intr, extrs, up_vec, centers[0])

    return render_c2ws

def get_camerapath(mode, c2ws, **kwargs):
    # mode = args.camera_path
    # kwargs list
    # numviews = args.num_views
    # view_ids = args.camera_inds.split(',')
    # view_ids = view_ids.split(',')
    # view_ids = [int(v) for v in view_ids]
    if mode == 'spiral':
        # Need num_views
        render_c2ws = c2w_track_spiral(c2ws, **kwargs)
    elif mode == 'spherical_spiral':
        # Need view_ids, debug?
        render_c2ws = c2w_track_sphericalspiral(c2ws, **kwargs)
    elif mode == 'small_circle':
        # Need view_ids, debug?
        render_c2ws = c2w_track_smallcircle(c2ws, **kwargs)
    elif mode == 'interpolation':
        # Need num_views
        render_c2ws = c2w_track_interpolation(c2ws, **kwargs)
    elif mode == 'great_circle':
        # Need view01
        render_c2ws = c2w_track_greatcircle(c2ws, **kwargs)
    elif mode == 'camviews':
        render_c2ws = 1
    else:
        raise RuntimeError(
            "Please choose render type between [spiral, interpolation, small_circle, great_circle, spherical_spiral]")

    return render_c2ws

def main_funciton(args, write_to):
    """
    Implemented Camera Paths
    spiral, interpolation, small_circle, great_circle, spherical_spiral
    """
    io_util.cond_mkdir(write_to)

    model, trainer, render_kwargs_train, render_kwargs_test, render_fn = get_model(args)
    if args.load_pt is None:
        # automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(args.training.exp_dir, 'ckpts'))[-1]
    else:
        ckpt_file = args.load_pt

    ## Load Model
    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=args.device)
    model.load_state_dict(state_dict['model'])
    model.to(args.device)

    ## Load Data
    from dataio import get_data
    dataset = get_data(args, downscale=args.downscale)
    subjectdata = dataset.subjects_data[args.subjectid]
    c2ws = subjectdata['c2w_all']
    intrinsics = subjectdata['intrinsics_all']

    ## Compute specified camera paths
    log.info("=> Camera path: {}".format(args.camera_path))
    campath_kwargs = {'numviews':args.visualization.numviews, 'view_ids':args.visualization.view_ids, 'view01':args.visualization.view01,
                      'debug':args.debug, 'intrinsics':intrinsics}
    render_c2ws = get_camerapath(args.camera_path, c2ws, **campath_kwargs)

    rgb_imgs = []
    depth_imgs = []
    normal_imgs = []
    # save mesh render images
    mesh_imgs = []
    render_kwargs_test['rayschunk'] = args.visualization.rayschunk

    do_render_mesh = args.mesh_path is not None
    if do_render_mesh:
        import open3d as o3d
        log.info("=> Load mesh: {}".format(args.render_mesh))
        geometry = o3d.io.read_triangle_mesh(args.render_mesh)
        geometry.compute_vertex_normals()
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=W, height=H, visible=args.debug)
        ctrl = vis.get_view_control()
        vis.add_geometry(geometry)
        cam = ctrl.convert_to_pinhole_camera_parameters()
        intr = intrinsics.data.cpu().numpy()
        # cam.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], intr[0,2], intr[1,2])
        cam.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], W/2-0.5, H/2-0.5)
        ctrl.convert_from_pinhole_camera_parameters(cam)


if __name__ == "__main__":
    # parser = io_util.create_args_parser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config_test/neus_nomask.yaml', help='Path to config file.')
    parser.add_argument('--resume_dir', type=str, default=None, help='Just for match to protocol.')
    parser.add_argument("--mesh_path", type=str, default=None, help='the mesh ply file to be rendered')
    parser.add_argument("--device", type=str, default='cuda', help='render device')
    parser.add_argument("--downscale", type=float, default=4)
    parser.add_argument("--rayschunk", type=int, default=1048)
    parser.add_argument("--subjectid", type=int, default=363)
    parser.add_argument("--num_views", type=int, default=200)
    parser.add_argument("--debug", action='store_true', help='Whether enable debuging camera path', default=False)
    ##
    parser.add_argument("--mesh_file", type=str, default="../logs/neusseg_nomask_jungwoo_exp_b_7/meshes/00300000.ply")
    parser.add_argument("--sphere_radius", type=float, default=3.0)
    parser.add_argument("--backface",action='store_true', help='render show back face')
    args = parser.parse_args()