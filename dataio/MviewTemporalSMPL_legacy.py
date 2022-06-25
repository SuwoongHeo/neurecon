import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from pytorch3d.transforms import so3_exp_map
from utils.io_util import load_mask, load_rgb, glob_imgs, get_img_paths, undistort_image
from utils.rend_util import rot_to_quat, load_K_Rt_from_P
from utils.texture_util import VertColor2UVMapper
from utils.geometry import Mesh

from zju_smpl.body_model import SMPLlayer
from zju_smpl.lbs import batch_rodrigues

asset_dir = os.path.dirname(os.path.abspath(__file__))+"/../assets/"
class SceneDataset(torch.utils.data.Dataset):
    # NOTE: jianfei: modified from IDR.   https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
    # Note: swheo : Modified for multiview temporal images, referring
    # neuralbody  https://github.com/zju3dv/neuralbody/blob/master/lib/datasets/light_stage/multi_view_dataset.py
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""
    # todo modify it to use multiple persons
    def __init__(self,
                 data_dir,
                 downscale=1.,   # [H, W]
                 cam_file=None,
                 subjects=[],
                 scale_radius=-1,
                 uv_size=256,
                 frame_int=1,
                 frame_max_num=300,
                 n_views=-1,
                 smpl_type='smpl',
                 maskbkg=False,
                 loadpts=False):

        assert os.path.exists(data_dir), "Data directory is empty"
        self.maskbkg = maskbkg
        self.instance_dir = data_dir
        self.downscale = downscale
        self.subjects = subjects

        image_dir = '{0}/images'.format(self.instance_dir)
        image_paths = get_img_paths(image_dir)
        image_paths = image_paths[0:min(len(image_paths), (frame_int*frame_max_num)):frame_int]
        mask_dir = '{0}/masks'.format(self.instance_dir)
        mask_paths = get_img_paths(mask_dir)
        mask_paths = mask_paths[0:min(len(mask_paths), (frame_int*frame_max_num)):frame_int]
        # smpl_dir = "{0}/output/{1}/smpl".format(self.instance_dir, smpl_type)
        # smpl_paths = sorted([os.path.join(smpl_dir,name) for name in os.listdir(smpl_dir) if name.endswith('.json')])
        # smpl_paths = smpl_paths[0:min(len(smpl_paths), (frame_int*frame_max_num)):frame_int]

        self.n_frames = len(image_paths)
        # views = np.random.randint(0, len(image_paths[0]), n_views) if n_views>0 else np.arange(0, len(image_paths[0])).astype(np.int64)
        self.views = np.random.randint(0, len(image_paths[0]), n_views) if n_views>0 else np.arange(0, len(image_paths[0])).astype(np.int64)
        self.n_views = len(self.views)
        # determine width, height
        self.downscale = downscale
        tmp_rgb = load_rgb(image_paths[0][0], downscale)
        _, self.H, self.W = tmp_rgb.shape

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        # normalization(scale) matrix :
        # normalize cameras such that the visual hull of the observed object is approximately inside the unit sphere
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.views]
        # P = K[R|t] , world_mats = concat(P, [0,0,0,1], axis=0)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.views]
        dist_mats = [camera_dict['dist_mat_%d' % idx].astype(np.float32) for idx in self.views]

        self.dists_all = []
        self.intrinsics_all = []
        self.c2w_all = []
        cam_center_norms = []
        for scale_mat, world_mat, dist_mat in zip(scale_mats, world_mats, dist_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(P)
            cam_center_norms.append(np.linalg.norm(pose[:3,3]))
            # pose[:3,:3] = -pose[:3,:3]
            # downscale intrinsics
            intrinsics[0, 2] /= downscale
            intrinsics[1, 2] /= downscale
            intrinsics[0, 0] /= downscale
            intrinsics[1, 1] /= downscale
            # intrinsics[0, 1] /= downscale # skew is a ratio, do not scale

            self.dists_all.append(torch.from_numpy(dist_mat).float())
            # self.intrinsics_all_np.append(intrinsics) #todo possibly performance bottleneck
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.c2w_all.append(torch.from_numpy(pose).float())

        max_cam_norm = max(cam_center_norms)
        if scale_radius > 0:
            for i in range(len(self.c2w_all)):
                self.c2w_all[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)
        # # Sanity check
        # if loadpts:
        #     self.pts = np.load(os.path.join(data_dir, 'vertices', '0.npy'))
        #     self.pts_scaled = np.concatenate([self.pts, np.ones((self.pts.shape[0],1))], axis=-1)\
        #                       @np.linalg.inv(scale_mats[0]).T
        #     # scale_mat_ = scale_mats[0].copy()
        #     # scale_mat_[:3,-1] = -scale_mat_[:3,-1]
        #     # self.pts_scaled = np.concatenate([self.pts, np.ones((self.pts.shape[0], 1))], axis=-1) \
        #     #                   @ scale_mat_.T
        #     mult = (scale_radius / max_cam_norm / 1.1) if scale_radius > 0 else 1
        #     self.pts_scaled = self.pts_scaled[:, :3] * mult
        scale_mat = np.linalg.inv(scale_mats[0]).T
        self.scale_mat = scale_mat*(scale_radius / max_cam_norm / 1.1) if scale_radius > 0 else scale_mat
        self.rgb_images = np.array([
            np.array(image_path)[self.views]
            for image_path in image_paths
        ]).ravel()
        self.masks = np.array([
            np.array(mask_path)[self.views]
            for mask_path in mask_paths
        ]).ravel()
        self.cam_inds = np.tile(self.views, self.n_frames)

        self.smpl = SMPLlayer(os.path.join(asset_dir, 'smpl', smpl_type),
                              model_type=smpl_type,
                              gender='neutral')

        mesh = Mesh(file_name=os.path.join(asset_dir, 'smpl', smpl_type, 'smpl_uv.obj'))
        uvmask = torch.from_numpy(load_rgb(os.path.join(asset_dir, 'smpl', smpl_type, 'smpl_uv.png'),
                                           downscale=1024./uv_size, anti_aliasing=False, order=0))
        uvmask = uvmask[-1]

        self.Renderer = VertColor2UVMapper(mesh.uv, mesh.faces_uv, uvmask[None], find_border=True)
        self.faces = mesh.faces

        foo = 1

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        # uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        # uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        # uv = uv.reshape(2, -1).transpose(1, 0)
        cam_ind = self.cam_inds[idx]
        K = self.intrinsics_all[cam_ind]
        D = self.dists_all[cam_ind].T
        img = torch.from_numpy(load_rgb(self.rgb_images[idx], self.downscale)).to(K.device) #todo check, device? cpu? cuda:#?
        img = undistort_image(img[None,...], K[None,:3,:3], D, mode='bilinear')[0,...]
        mask = torch.from_numpy(load_mask(self.masks[idx], self.downscale, False)).to(K.device)
        mask = undistort_image(mask[None,None,...].type(torch.float32), K[None,:3,:3], D, mode='nearest')[0,...].type(torch.long)

        object_mask = mask>0.
        ind = int(self.rgb_images[idx].split('/')[-1].split('.')[0])
        # vertices = torch.from_numpy(np.load(os.path.join(self.instance_dir, 'vertices', f'{ind}.npy'))).to(K.device)
        smpl_params = np.load(os.path.join(self.instance_dir, 'params', f'{ind}.npy'), allow_pickle=True).item()

        for key, val in smpl_params.items():
            smpl_params[key] = torch.from_numpy(val).float().to(K.device)
        if self.maskbkg:
            img[torch.tile(object_mask, dims=(3, 1, 1)) == 0] = 0
        # todo, smpl canonical space for pose feature computation
        vertices_world = self.smpl(return_verts=True,
                        return_tensor=True,
                        new_params=True,
                        **smpl_params)[0]
        R = batch_rodrigues(smpl_params['Rh'])[0]
        T = smpl_params['Th']

        vertices_can = (vertices_world - T)@R
        pelvis = self.smpl.J_regressor[0,:]@vertices_can
        vertices_can -= pelvis[None]
        cbuvmap = self.Renderer(vertices_can, self.faces)
        vertices_world = (torch.concat([vertices_world, torch.ones((vertices_world.shape[0], 1), device=vertices_world.device)], dim=-1) \
         @ self.scale_mat)[..., :3]
        sample = {
            "object_mask": object_mask,
            "intrinsics": K,
            "vertices": vertices_world,
            "vertices_can": vertices_can,
            "smpl_params": smpl_params,
            "cbuvmap": cbuvmap,
            "garment_id": 0 #todo, extend it when it comes to multiple persons
        }

        ground_truth = {
            "rgb": img.reshape(3,-1),
            "segm": mask.reshape(-1)
        }

        sample["c2w"] = self.c2w_all[cam_ind]

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

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=True):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.views]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.views]

        c2w_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = load_K_Rt_from_P(P)
            c2w_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in c2w_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.views]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.views]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = load_K_Rt_from_P(P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    dataset = SceneDataset('/ssd2/swheo/db/ZJU_MOCAP/LightStage/363')#, scale_radius=5) # /ssd2/swheo/db/DTU/scan65
    from tools.vis_ray import visualize_frustum
    ax = visualize_frustum(dataset, [], obj_bounding_radius=1., show=False)

    # Test closet surface point search
    from zju_smpl.body_model import SMPLlayer
    from utils.geometry import Mesh, project2closest_face, texcoord2imcoord
    data = dataset.__getitem__(0)
    # pose model and save vertices
    vertices = data[1]['vertices']
    # verts = smpl.v_template.numpy()
    # faces = smpl.faces
    # faces = mesh_.faces_list()[0].numpy()
    # mesh = Mesh(vertices=smpl.v_template, faces=smpl.faces)
    # mesh = IO().load_mesh(smpl_dir)
    # mesh = Mesh(vertices=verts, faces=faces)
    mesh = Mesh(file_name= '../assets/smpl/smpl/smpl_uv.obj')
    uvmap = torch.from_numpy(load_rgb('../assets/smpl/smpl/smpl_uv.png', downscale=4))
    uvmask = uvmap[-1]
    uvmap = uvmap[:-1]*uvmask.unsqueeze(0)
    mesh.update_vertices(vertices)
    numquery = 1 #256*1024
    bbox = torch.stack([mesh.vertices.min(dim=0).values, mesh.vertices.max(dim=0).values])
    # query : [(B), N_rays, N_samples+N_importance, 3]
    query = bbox[0,:] + torch.rand((numquery, 3))*(bbox[-1,:] - bbox[0,:])
    query = query[None, None, ...] # Make batch and single ray

    vnear, st, idxs = project2closest_face(query.to(mesh.device), mesh)

    uv = mesh.get_uv_from_st(st, idxs)
    xy = texcoord2imcoord(uv, uvmap.shape[1], uvmap.shape[2])
    pp = vnear.view(-1,3).cpu().numpy()
    qq = query.view(-1, 3).cpu().numpy()
    x, y, z, = mesh.vertices.cpu().chunk(3, -1)
    ax.plot_trisurf(x.numpy()[:, 0], y.numpy()[:, 0], z.numpy()[:, 0],
                    triangles=mesh.faces.numpy(), alpha=0.5, edgecolor=[0, 0, 0])
    ax.scatter3D(pp[:, 0], pp[:, 1], pp[:, 2], s=20, c="red")
    ax.scatter3D(qq[:, 0], qq[:, 1], qq[:, 2], s=20, c="red")
    for qqq, ppp in zip(qq, pp):
        ax.plot([qqq[0], ppp[0]], [qqq[1], ppp[1]], [qqq[2], ppp[2]], c="blue")
    import matplotlib.pyplot as plt
    plt.show()
    foo = 1