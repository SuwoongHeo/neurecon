import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

matplotlib.use('TkAgg')
from utils.rend_util import get_rays, get_bound_rays, near_far_from_sphere, get_sphere_intersection
from dataio.DTU import SceneDataset
from dataio.MviewTemporalSMPL import SceneDataset as SceneDataset_MView

# Same
def create_frustum_model(center, pts, draw_frame_axis=False):
    near_p = pts[0] # Near plane
    far_p = pts[1] # Far plane

    # draw near plane
    X_near_plane = np.ones((4,5))
    X_near_plane[0:3, 0] = near_p[2] # [-nwidth, nheight, near]
    X_near_plane[0:3, 1] = near_p[3] # [nwidth, nheight, near]
    X_near_plane[0:3, 2] = near_p[1] # [nwidth, -nheight, near]
    X_near_plane[0:3, 3] = near_p[0] # [-nwidth, -nheight, near]
    X_near_plane[0:3, 4] = near_p[2] # [-nwidth, nheight, near]

    # draw far plane
    X_far_plane = np.ones((4, 5))
    X_far_plane[0:3, 0] = far_p[2] # [-width, height, far]
    X_far_plane[0:3, 1] = far_p[3] # [width, height, far]
    X_far_plane[0:3, 2] = far_p[1] # [width, -height, far]
    X_far_plane[0:3, 3] = far_p[0] # [-width, -height, far]
    X_far_plane[0:3, 4] = far_p[2] # [-width, height, far]

    # draw frustum
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = center
    X_center1[0:3, 1] = far_p[2] # [-width, height, far]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = center
    X_center2[0:3, 1] = far_p[3] # [width, height, far]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = center
    X_center3[0:3, 1] = far_p[1] # [width, -height, far]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = center
    X_center4[0:3, 1] = far_p[0] # [-width, -height, far]

    # draw camera frame axis
    axlen = np.max([np.linalg.norm(near_p[1] - near_p[0]), np.linalg.norm(near_p[2] - near_p[0])])
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [axlen/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, axlen/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, axlen/2]

    if draw_frame_axis:
        return [X_near_plane, X_far_plane, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_near_plane, X_far_plane, X_center1, X_center2, X_center3, X_center4]

def visualize_frustum(dataset, camera_matrix,
                      mesh=None, obj_bounding_radius=1., N_view=-1, num_cams=None, HW=None,
                      use_sphereinter=False, show=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable
    from matplotlib import cm

    import torch.nn.functional as F

    n_views = num_cams if num_cams is not None else dataset.n_views
    H, W = HW if HW is not None else (dataset.H, dataset.W)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ## ax.set_aspect("equal")
    ax.set_aspect("auto")
    if mesh != None:
        ax.plot_trisurf(mesh['x'], mesh['y'], mesh['z'], triangles=mesh['f'], alpha=0.5)
    cm_subsection = np.linspace(0.0, 1.0, n_views)
    colors = [cm.jet(x) for x in cm_subsection]

    min_values = np.inf
    max_values = -np.inf

    # Randomly choose views to be used
    views = np.sort(np.random.choice(n_views, size=N_view, replace=False)) if N_view>0 else np.arange(n_views)
    # Compute bounding rays

    for i in range(n_views):
        _, model_input, _ = dataset[i]
        intrinsics = model_input["intrinsics"][None, ...]
        c2w = model_input['c2w'][None, ...]
        rays_o, rays_d, select_inds = get_bound_rays(c2w, intrinsics, H, W)
        rays_d = F.normalize(rays_d, dim=-1)
        # rays_o = rays_o.data.squeeze(0).cpu().numpy()
        # rays_d = rays_d.data.squeeze(0).cpu().numpy()
        near, far = near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
        if use_sphereinter:
            near = torch.ones([rays_o.shape[0], rays_o.shape[1], 1]).to(rays_o.device)
            _, far, mask_intersect = get_sphere_intersection(rays_o, rays_d, r=obj_bounding_radius)
        if i not in views:
            near, far = 0.0*near, 0.3*(far/far.max())
        rays_o = rays_o.data.cpu().numpy()
        rays_d = rays_d.data.cpu().numpy()
        # pts_ = np.concatenate([rays_o + rays_d, rays_o+1.2*rays_d], axis=0)
        pts_ = np.concatenate([rays_o + near.cpu().numpy() * rays_d, rays_o + far.cpu().numpy() * rays_d], axis=0)
        center = rays_o[0,0,:]
        frustum_i = create_frustum_model(center, pts_)
        for idx, prim in enumerate(frustum_i):
            if idx > 1:
                ax.plot3D(prim[0, :], prim[1, :], prim[2, :], '--', color=colors[i], alpha=.8, linewidth=0.5)
            elif idx==0:
                ax.plot3D(prim[0, :], prim[1, :], prim[2, :], '-.', color=colors[i], alpha=.8, linewidth=1.5)
            else:
                ax.plot3D(prim[0, :], prim[1, :], prim[2, :], color=colors[i], alpha=.8, linewidth=1.)
            min_values = np.minimum(min_values, prim[0:3, :].min(1))
            max_values = np.maximum(max_values, prim[0:3, :].max(1))
        ax.text(center[0], center[1], center[2], "{}".format(i), color=colors[i])

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Extrinsic Parameters Visualization')

    if show:
        plt.show()

    return ax

def plot_rays(rays_o: np.ndarray, rays_d: np.ndarray, ax):
    # TODO: automatic reducing number of rays
    XYZUVW = np.concatenate([rays_o, rays_d], axis=-1)
    X, Y, Z, U, V, W = np.transpose(XYZUVW)
    # X2 = X+U
    # Y2 = Y+V
    # Z2 = Z+W
    # x_max = max(np.max(X), np.max(X2))
    # x_min = min(np.min(X), np.min(X2))
    # y_max = max(np.max(Y), np.max(Y2))
    # y_min = min(np.min(Y), np.min(Y2))
    # z_max = max(np.max(Z), np.max(Z2))
    # z_min = min(np.min(Z), np.min(Z2))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_zlim(z_min, z_max)

    return ax

if __name__=="__main__":
    dataset = SceneDataset(False, '/ssd2/swheo/dev/code/neurecon/data/zju_test', cam_file="cameras.npz", loadpts=False, scale_radius=3) #'../data/junghang_2/colmap/processed'
    # dataset = SceneDataset_MView('/ssd2/swheo/db/ZJU_MOCAP/LightStage/363', scale_radius=3) # /ssd2/swheo/db/DTU/scan65
    # dataset = SceneDataset(False, '../data/jungwoo_wmask/colmap/processed', cam_file="cameras.npz", loadpts=True, scale_radius=3.)
    ax = visualize_frustum(dataset, [], obj_bounding_radius=3.)
    foo = 13
    # Load
    if False:
        meshpath = '../logs/neus_nomask_junghang2_mar_ii/meshes/simplified.ply'
        # meshpath = '../logs/neus_nomask_jungwoo_exp_a_7/meshes/simplified.ply'
        import open3d as o3d
        geometry = o3d.io.read_triangle_mesh(meshpath)
        face = np.array(geometry.triangles)
        v = np.array(geometry.vertices)
        mesh_info = {'x':v[:,0], 'y':v[:,1], 'z':v[:,2], 'f':face}

        if True:
            import os
            reproj_path = '../logs/neus_nomask_junghang2_mar_ii/debug'
            # reproj_path = '../logs/neus_nomask_jungwoo_exp_a_7/debug'
            os.makedirs(reproj_path, exist_ok=True)
            for idx in range(len(dataset.rgb_images)):
                xy = dataset.intrinsics_all[idx].numpy() @ np.linalg.inv(dataset.c2w_all[idx]) @ np.concatenate(
                    [v, np.ones((v.shape[0], 1))], axis=-1).T
                xy = xy[:2, :] / xy[2, :]
                plt.figure()
                plt.imshow(dataset.rgb_images[idx].reshape((dataset.H, dataset.W, 3)))
                plt.scatter(xy[0, :], xy[1, :], s=1)
                plt.savefig(os.path.join(reproj_path, f'img_{idx:03}.png'))
        ax = visualize_frustum(dataset, [], obj_bounding_radius=1., mesh=mesh_info)
        ax.scatter3D(dataset.pts_scaled[:, 0], dataset.pts_scaled[:, 1], dataset.pts_scaled[:, 2], s=0.5)
        foo = 1
