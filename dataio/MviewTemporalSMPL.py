import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from utils.io_util import load_mask, load_rgb, glob_imgs, get_img_paths, undistort_image
from utils.rend_util import rot_to_quat, load_K_Rt_from_P, get_2dmask_from_bbox
from utils.train_util import grid_sample
from utils.texture_util import VertColor2UVMapper
from utils.geometry import Mesh

from zju_smpl.body_model import SMPLlayer
from zju_smpl.lbs import batch_rodrigues

# For IGR style pretraining
from scipy.spatial import cKDTree

asset_dir = os.path.dirname(os.path.abspath(__file__))+"/../assets/"
class SceneDataset(torch.utils.data.Dataset):
    # NOTE: jianfei: modified from IDR.   https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
    # Note: swheo : Modified for multiview temporal images, referring
    # neuralbody  https://github.com/zju3dv/neuralbody/blob/master/lib/datasets/light_stage/multi_view_dataset.py
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""
    # todo pixel transform needed?
    # How about, if input n_views is a list, then use that only?
    def __init__(self,
                 data_dir,
                 downscale=1.,   # [H, W]
                 subjects=[],
                 scale_radius=-1,
                 uv_size=256,
                 start_frame=0,
                 end_frame=-1,
                 num_frame=-1,
                 select_frame='uniform',
                 views=[],
                 smpl_type='smpl',
                 maskbkg=False,
                 **kwargs):

        assert os.path.exists(data_dir), "Data directory is empty"
        self.maskbkg = maskbkg
        self.instance_dir = data_dir
        self.downscale = downscale
        self.subjects = subjects
        self.smpl = SMPLlayer(os.path.join(asset_dir, 'smpl', smpl_type),
                              model_type=smpl_type,
                              gender='neutral')
        mesh = Mesh(file_name=os.path.join(asset_dir, 'smpl', smpl_type, 'smpl_uv.obj'))

        self.subjects_data = {name:dict() for name in subjects}
        self.rgb_images = np.array([])
        self.masks = np.array([])
        for i, subject in enumerate(subjects):
            self.subjects_data[subject]['subject_id'] = i
            image_dir = f'{self.instance_dir}/{subject}/images'
            image_paths = get_img_paths(image_dir)
            frame_max = len(image_paths) if end_frame == -1 else end_frame
            mask_dir = f'{self.instance_dir}/{subject}/masks'
            mask_paths = get_img_paths(mask_dir)

            cam_file = f'{self.instance_dir}/{subject}/cameras.npz'
            self.subjects_data[subject]['cam_file'] = cam_file

            assert select_frame in ['uniform', 'random'], 'only uniform or random frame selections are supproted'
            frame_interval = max(int((start_frame - frame_max + 1) / num_frame), 1)
            frame_inds = np.arange(start_frame, frame_max, frame_interval)[
                         :num_frame] if select_frame == 'uniform' else \
                np.sort(np.random.choice(np.arange(start_frame, frame_max), size=max(num_frame, 1), replace=False))
            print(f"Subject {subject}, start_fram {frame_inds[0]}, end_frame {frame_inds[-1]}")
            print(f"uniform samples, frame_interval {frame_interval}") if select_frame == 'uniform' else \
                print(f"random samples, num_samples {num_frame} ")
            self.subjects_data[subject]['frame_inds'] = frame_inds

            image_paths = [image_paths[ind] for ind in frame_inds]
            mask_paths = [mask_paths[ind] for ind in frame_inds]

            all_views = np.arange(0, len(image_paths[0])) .astype(np.int64)
            views_ = np.array(views).astype(np.int64)-1 if len(views) >0 else all_views
            # self.n_views = len(views) #temporary variable for camera visualization
            self.subjects_data[subject]['views'] = views_

            tmp_rgb = load_rgb(image_paths[0][0], downscale)
            _, H, W = tmp_rgb.shape
            self.subjects_data[subject]['H'] = H
            self.subjects_data[subject]['W'] = W

            camera_dict = np.load(cam_file)
            # normalization(scale) matrix :
            # normalize cameras such that the visual hull of the observed object is approximately inside the unit sphere
            scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in all_views]
            # P = K[R|t] , world_mats = concat(P, [0,0,0,1], axis=0)
            world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in all_views]
            dist_mats = [camera_dict['dist_mat_%d' % idx].astype(np.float32) for idx in all_views]

            dists_all = []
            intrinsics_all = []
            c2w_all = []
            cam_center_norms = []
            for scale_mat, world_mat, dist_mat in zip(scale_mats, world_mats, dist_mats):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(P)
                cam_center_norms.append(np.linalg.norm(pose[:3, 3]))
                # pose[:3,:3] = -pose[:3,:3]
                # downscale intrinsics
                intrinsics[0, 2] /= downscale
                intrinsics[1, 2] /= downscale
                intrinsics[0, 0] /= downscale
                intrinsics[1, 1] /= downscale
                # intrinsics[0, 1] /= downscale # skew is a ratio, do not scale

                dists_all.append(torch.from_numpy(dist_mat).float())
                intrinsics_all.append(torch.from_numpy(intrinsics).float())
                c2w_all.append(torch.from_numpy(pose).float())

            self.subjects_data[subject]['dists_all'] = dists_all
            self.subjects_data[subject]['intrinsics_all'] = intrinsics_all

            max_cam_norm = max(cam_center_norms)
            if scale_radius > 0:
                for i in range(len(c2w_all)):
                    c2w_all[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)
            self.subjects_data[subject]['c2w_all'] = c2w_all
            scale_mat = np.linalg.inv(scale_mats[0]).T
            scale_mat = scale_mat * (scale_radius / max_cam_norm / 1.1) if scale_radius > 0 else scale_mat
            self.subjects_data[subject]['scale_mat'] = torch.from_numpy(scale_mat).float()

            rgb_images = np.array([
                np.array(image_path)[views_]
                for image_path in image_paths
            ]).ravel()
            self.rgb_images = np.concatenate([self.rgb_images, rgb_images])
            masks = np.array([
                np.array(mask_path)[views_]
                for mask_path in mask_paths
            ]).ravel()
            self.masks = np.concatenate([self.masks, masks])
            # cam_inds = np.tile(views, num_frame)

            # Creates and registers canonical body shape parameters

            smpl_params = np.load(os.path.join(self.instance_dir, subject, 'params', '0.npy'),
                                  allow_pickle=True).item()

            # Make it Dae (Hanja) shape
            pose_canonical = np.zeros_like(smpl_params['poses'])
            pose_canonical[0,5] = np.pi/ 6.
            pose_canonical[0,8] = -np.pi / 6.

            vertices_tposed, transforms_world = self.smpl(return_verts=True,
                                        return_tensor=True,
                                        new_params=True,
                                        return_transforms=True,
                                        poses=pose_canonical, shapes=smpl_params['shapes'])
            # Move to smpl tpose to zero centered at pelvis
            # pelvis = self.smpl.J_regressor[0, :] @ vertices_tposed
            # vertices_tposed -= pelvis[None]
            mesh.update_vertices(vertices_tposed[0])
            self.subjects_data[subject]['tposeInfo'] = {'vertices': vertices_tposed[0], 'B': mesh.triangles[:,0],
                                                        'E0': mesh.edges[:,0], 'E1': -mesh.edges[:,1],
                                                        'tanframe': mesh.faces_tanframe,
                                                        'tanframe_inv': mesh.faces_tanframe.transpose(-1, -2)}
                                                        #'tanframe_inv':torch.linalg.inv(mesh.faces_tanframe)}
            self.subjects_data[subject]['Acan_inv'] = torch.from_numpy(np.linalg.inv(transforms_world))

        uvmask = torch.from_numpy(load_rgb(os.path.join(asset_dir, 'smpl', smpl_type, 'smpl_uv.png'),
                                           downscale=1024./uv_size, anti_aliasing=False, order=0))
        uvmask = uvmask[-1]

        self.Renderer = VertColor2UVMapper(mesh.uv, mesh.faces_uv, uvmask[None], find_border=True)
        self.faces = mesh.faces

        # Computing the pretraining parameters
        # Precompute local deviation of poitns within 50-nearest neighbors
        ptree_neutral = cKDTree(mesh.vertices.numpy())
        sigma_set = []
        for p in np.array_split(mesh.vertices.numpy(), 100, axis=0):
            d = ptree_neutral.query(p, 50 + 1)
            sigma_set.append(d[0][:, -1])
        self.local_sigma = np.concatenate(sigma_set)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        img_path = self.rgb_images[idx]
        mask_path = self.masks[idx]
        subject = img_path.split('/')[-4]
        cam_ind = int(img_path.split('/')[-2])-1
        frameind = int(img_path.split('/')[-1].split('.')[0])
        K = self.subjects_data[subject]['intrinsics_all'][cam_ind]
        D = self.subjects_data[subject]['dists_all'][cam_ind].T
        img = torch.from_numpy(load_rgb(img_path, self.downscale)).to(
            K.device)
        mask = torch.from_numpy(load_mask(mask_path, self.downscale, False)).to(K.device)
        if torch.any(D != 0.).item():
            img = undistort_image(img[None, ...], K[None, :3, :3], D, mode='bilinear')[0, ...]
            mask = undistort_image(mask[None, None, ...].type(torch.float32), K[None, :3, :3], D, mode='nearest')[
            0, ...].type(torch.long)

        # todo dilate mask?
        object_mask = mask>0.
        # vertices = torch.from_numpy(np.load(os.path.join(self.instance_dir, 'vertices', f'{ind}.npy'))).to(K.device)
        smpl_params = np.load(os.path.join(self.instance_dir, subject, 'params', f'{frameind}.npy'), allow_pickle=True).item()

        for key, val in smpl_params.items():
            smpl_params[key] = torch.from_numpy(val).float().to(K.device)

        if self.maskbkg:
            img[torch.tile(object_mask, dims=(3, 1, 1)) == 0] = 0

        vertices_world,transforms_world = self.smpl(return_verts=True,
                        return_tensor=True,
                        new_params=True,
                        return_transforms=True,
                        **smpl_params)
        vertices_world = vertices_world[0]
        R = batch_rodrigues(smpl_params['Rh'])[0]
        T = smpl_params['Th']
        c2w = self.subjects_data[subject]['c2w_all'][cam_ind]

        # Move to smpl object(canonical)  space
        vertices_can = (vertices_world - T)@R
        pelvis = self.smpl.J_regressor[0,:]@vertices_can
        vertices_can -= pelvis[None]
        cbuvmap = self.Renderer(vertices_can, self.faces)

        Tglo = torch.cat([F.pad(R, [0, 1, 0, 0]), F.pad(-T @ R, [0, 1], value=1.)], dim=0)
        transformInfo = {'alignMat':torch.linalg.inv(self.subjects_data[subject]['scale_mat'])@Tglo,
                         'A': transforms_world@self.subjects_data[subject]['Acan_inv'], 'W':self.smpl.weights}

        # Match world vertices to the world space of cameras
        vertices_world = (torch.concat([vertices_world, torch.ones((vertices_world.shape[0], 1), device=vertices_world.device)], dim=-1) \
         @ self.subjects_data[subject]['scale_mat'])[..., :3]

        sample = {
            "object_mask": object_mask, #object_mask.reshape(-1),
            "intrinsics": K,
            "vertices": vertices_world,
            "vertices_can": vertices_can,
            "tposeInfo": self.subjects_data[subject]['tposeInfo'],
            "transformInfo": transformInfo,
            "smpl_params": smpl_params,
            "cbuvmap": cbuvmap.permute(2, 0, 1),
            "subject_id": self.subjects_data[subject]['subject_id'],
            "subject": subject,
            "tag": f"s{subject}c{cam_ind}f{frameind}"
        }

        ground_truth = {
            "rgb": img.reshape(3,-1).transpose(1,0),
            "segm": mask.reshape(-1)
        }

        sample["c2w"] = c2w

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)


def plotly_viscorres3D(vnear, query, pp_color, vertices, faces, faces_tanframe=None):
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"

    pp = vnear.view(-1, 3).cpu().numpy()
    qq = query.view(-1, 3).cpu().numpy()
    x, y, z, = vertices.cpu().chunk(3, -1)
    plmesh = go.Mesh3d(x=x.numpy()[:, 0], y=y.numpy()[:, 0], z=z.numpy()[:, 0], i=faces.numpy()[:, 0],
                       j=faces.numpy()[:, 1], k=faces.numpy()[:, 2], color='grey', opacity=.6,
                       lighting=dict(ambient=0.2, diffuse=0.8), lightposition=dict(x=0, y=0, z=-1))
    pp_marker = go.Scatter3d(x=pp[:, 0], y=pp[:, 1], z=pp[:, 2], marker=go.scatter3d.Marker(size=3, color=pp_color),
                             mode='markers')
    qq_marker = go.Scatter3d(x=qq[:, 0], y=qq[:, 1], z=qq[:, 2], marker=go.scatter3d.Marker(size=3, color=pp_color),
                             mode='markers')
    x_lines, y_lines, z_lines = list(), list(), list()
    for qqq, ppp in zip(qq, pp):
        x_lines.extend([qqq[0], ppp[0], None])
        y_lines.extend([qqq[1], ppp[1], None])
        z_lines.extend([qqq[2], ppp[2], None])
    lines = go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines')
    frames = []
    if faces_tanframe is not None:
        frames_np = faces_tanframe.view(-1, 3, 3).cpu().numpy()
        for ax in range(frames_np.shape[-2]):
            # x', y', z'
            x_lines, y_lines, z_lines = list(), list(), list()
            ff = pp + frames_np[:, ax, :]
            for fff, ppp in zip(ff, pp):
                x_lines.extend([fff[0], ppp[0], None])
                y_lines.extend([fff[1], ppp[1], None])
                z_lines.extend([fff[2], ppp[2], None])
            frames.append(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines'))
    invisible_scale = go.Scatter3d(name="", visible=True, showlegend=False, opacity=0, hoverinfo='none',
                                   x=[-1.2, 1.2], y=[-1.2, 1.2], z=[-1.2, 1.2])
    fig = go.Figure(data=[invisible_scale, plmesh, pp_marker, qq_marker, lines, *frames])
    # fig['layout']['scene']['aspectmode'] = "data"
    # fig['layout']['scene']['xaxis']['range'] = [-.5, .5]
    # fig['layout']['scene']['yaxis']['range'] = [-1., 1.]
    # fig['layout']['scene']['zaxis']['range'] = [-.5, .5]
    fig.show()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    dataset = SceneDataset('/ssd2/swheo/db/ZJU_MOCAP/LightStage', subjects=['363'], views=[], num_frame=300)#, scale_radius=5) # /ssd2/swheo/db/DTU/scan65
    # Test closet surface point search
    from zju_smpl.body_model import SMPLlayer
    from utils.geometry import Mesh, project2closest_face, texcoord2imcoord
    sanitycheck = False
    closestuv_check = True
    """
    Sanity check
    """
    if sanitycheck:
        import matplotlib.pyplot as plt
        from pytorch3d.io import IO
        from pytorch3d.structures import Meshes
        for ii, idx in enumerate(np.random.randint(0, len(dataset), 15)):
            data = dataset.__getitem__(idx)
            meshes = Meshes(verts=[data[1]['vertices_can']], faces=[dataset.faces])
            IO().save_mesh(data=meshes, path=os.path.join('./', 'sanitycheck', f"{ii}.obj"))
            ## pose model and save vertices
            # vertices = data[1]['vertices']
            # xy = data[1]['intrinsics']@torch.linalg.inv(data[1]['c2w'])@ \
            #      torch.concat([vertices, torch.ones((vertices.shape[0], 1), device=vertices.device)], dim=-1).T
            # xy = (xy[:2, :]/xy[2, :]).cpu().numpy()
            # img = data[2]['rgb'].reshape((data[1]['H'], data[1]['W'], -1)).cpu().numpy()
            # plt.figure()
            # plt.imshow(img)
            # plt.scatter(xy[0,:], xy[1,:], s=1)
            # plt.savefig(os.path.join('./', 'sanitycheck', f"{ii}.png"))
            # plt.close()
        import matplotlib.cm as colormap
        cmap = colormap.get_cmap('jet', 4)
        labels_cmap = torch.from_numpy(cmap(range(4))[:, :3])
        data = dataset.__getitem__(0)
        ddd = data[2]['segm'].cpu().numpy().reshape(data[1]['H'], data[1]['W'], -1)
        plt.imshow(labels_cmap[ddd].squeeze())
    if closestuv_check:
        # pose model and save vertices
        data = dataset.__getitem__(1)
        vertices = data[1]['vertices']
        mesh = Mesh(file_name= '../assets/smpl/smpl/smpl_uv.obj')
        vtemplate = dataset.smpl.v_template
        uvmap = torch.from_numpy(load_rgb('../assets/smpl/smpl/smpl_uv.png', downscale=4))
        uvmask = uvmap[-1]
        uvmap = uvmap[:-1]*uvmask.unsqueeze(0)
        # #todo
        # vertices = data[1]['tposeInfo']['vertices']
        mesh.update_vertices(vertices)
        numquery = 2 #256*1024
        numsample = 3
        c2w = data[1]['c2w']
        intrinsics = data[1]['intrinsics']
        H, W = data[1]['object_mask'].shape

        # near * (1 - _t) + far * _t
        # query : [(B), N_rays, N_samples+N_importance, 3]
        # query = bbox[0,:] + torch.rand((numquery, 3))*(bbox[-1,:] - bbox[0,:])
        # query = query[None, None, ...] # Make batch and single ray
        from utils import rend_util
        # Computed object bounding bbox
        margin = 0.1
        bbox = torch.stack([data[1]['vertices'].min(dim=0).values-margin, data[1]['vertices'].max(dim=0).values+margin])
        bbox_mask = rend_util.get_2dmask_from_bbox(bbox, intrinsics, c2w, H, W)
        # bbox = bbox.to(c2w.device)
        # bbox_mask = rend_util.get_2dmask_from_bbox(bbox, intrinsics, c2w, H, W)
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=numquery, mask=bbox_mask)
        rays_d = F.normalize(rays_d, dim=-1)
        near, far, valididx = rend_util.near_far_from_bbox(rays_o, rays_d, bbox)
        _t = torch.linspace(0, 1, numsample).float().to(rays_o.device)
        dquery = (near * (1 - _t) + far * _t)
        query = rays_o.unsqueeze(-2) + dquery.unsqueeze(-1) * rays_d.unsqueeze(-2)
        # query = torch.Tensor([-0.0078, -0.06779, -0.28070], device=query.device)[None,None,...]
        query = query[None].contiguous()
        #todo on smpl vertices debuging
        # sample_inds = torch.randint(vertices.shape[0], (50,), device=c2w.device)
        # pts_on = vertices[sample_inds, :][None,None,...]
        # pts_out_loc = pts_on + torch.randn_like(pts_on) * 0.1
        # pts_on = torch.cat([pts_on, pts_out_loc], dim=-2)
        # query = pts_on.contiguous()
        vnear, st, idxs = project2closest_face(query.to(mesh.device), mesh)

        # todo inverse skinning test
        A = data[1]['transformInfo']['A']
        W = data[1]['transformInfo']['W']
        alignMat = data[1]['transformInfo']['alignMat']
        # todo test canonical
        # alignMat = torch.eye(4, device=A.device).expand_as(data[1]['transformInfo']['alignMat']) # Optimize to T-pose and skin later
        # A = torch.eye(4, device=A.device).expand_as(A)

        # Move to smpl object(canonical)  space
        query_hom = torch.concat([query, torch.ones(query[...,0].shape, device=query.device).unsqueeze(-1)], dim=-1)
        query_can = (query_hom@alignMat)[...,:3]
        # dists_inv = 1./(torch.sum((query.view(-1, 1, 3) - vertices.view(1, -1, 3)) ** 2, dim=-1) + 1e-8)
        # W_q = (dists_inv/torch.sum(dists_inv, dim=-1).unsqueeze(-1)) @ W
        dists = torch.sum((query.view(-1, 1, 3) - vertices.view(1, -1, 3)) ** 2, dim=-1)
        beta = 3e-2
        W_q = (0.5*torch.exp(-torch.abs(dists)/beta)/beta) @ W
        WB = W[mesh.faces][idxs, 0]
        WE0 = (W[mesh.faces][:, 1, :] - W[mesh.faces][:, 0, :])[idxs]
        WE1 = (-W[mesh.faces][:, 0, :] + W[mesh.faces][:, 2, :])[idxs]
        W_q = (WB + st[..., 0].unsqueeze(-1) * WE0 + st[..., 1].unsqueeze(-1) * WE1).view(-1, 24)
        W_q = W_q/torch.sum(W_q+1e-8, dim=-1).unsqueeze(-1)
        T = (W_q[None].expand([1,-1,-1])@A.contiguous().view(1, W.shape[-1], 16)).view(1, -1, 4, 4)
        q_hom = torch.cat([query_can.view(-1,3), torch.ones_like(query_can.view(-1,3)[...,0])[...,None]], dim=-1)
        q_inv = torch.linalg.inv(T)@(q_hom.view(-1,4,1))
        q_inv = q_inv[...,:3, 0].view(query.shape)
        # q_inv = query_can
        uv = mesh.get_uv_from_st(st, idxs)
        xy = texcoord2imcoord(uv, 2, 2)
        xy_ = xy * 2. - 1.
        pp_color = grid_sample(uvmap[None], xy_).permute(0, 2, 3, 1).view(-1,3).cpu().numpy()
        tanframe = None # mesh.faces_tanframe[idxs]
        plotly_viscorres3D(vnear, query, pp_color, mesh.vertices, mesh.faces, faces_tanframe=tanframe)
        B = data[1]['tposeInfo']['B'][idxs] #mesh.triangles[idxs,0]
        E1 = data[1]['tposeInfo']['E1'][idxs] #mesh.triangles[idxs,0]
        E0 = data[1]['tposeInfo']['E0'][idxs] #mesh.triangles[idxs,0]
        q_diff = query - vnear
        q_tf = mesh.faces_tanframe[idxs] @ q_diff.unsqueeze(-1)
        vnear_tpose = B + st[...,0].unsqueeze(-1)*E0 + st[...,1].unsqueeze(-1)*E1
        q_tpose = (data[1]['tposeInfo']['tanframe'][idxs].transpose(-1,-2) @ q_tf)[...,0] + vnear_tpose
        tanframe = None # data[1]['tposeInfo']['tanframe'][idxs]
        plotly_viscorres3D(vnear_tpose, q_tpose, pp_color, data[1]['tposeInfo']['vertices'], mesh.faces, faces_tanframe=tanframe)
        plotly_viscorres3D(vnear_tpose, q_inv, pp_color, data[1]['tposeInfo']['vertices'], mesh.faces,
                           faces_tanframe=tanframe)

        # todo dont delete below, before making function for drawing camera on plotly
        # from tools.vis_ray import visualize_frustum
        # ax = visualize_frustum(dataset, [], obj_bounding_radius=1.,
        #                        num_cams = len(dataset.subjects_data['363']['views']),
        #                        HW= (dataset.subjects_data['363']['H'], dataset.subjects_data['363']['W']),
        #                        show=False)
        ## import matplotlib.pyplot as plt
        ## fig=plt.figure()
        ## ax = fig.gca(projection='3d')
        # ax.set_aspect("auto")
        # ax.plot_trisurf(x.numpy()[:, 0], y.numpy()[:, 0], z.numpy()[:, 0],
        #                 triangles=mesh.faces.numpy(), alpha=0.5, edgecolor=[0, 0, 0])
        # ax.scatter3D(pp[:, 0], pp[:, 1], pp[:, 2], s=20, c=pp_color)
        # ax.scatter3D(qq[:, 0], qq[:, 1], qq[:, 2], s=20, c=pp_color)
        # for qqq, ppp in zip(qq, pp):
        #     ax.plot([qqq[0], ppp[0]], [qqq[1], ppp[1]], [qqq[2], ppp[2]], c="blue")
        # # import matplotlib.pyplot as plt
        # plt.show()
        foo = 1