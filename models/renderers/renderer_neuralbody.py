from utils import rend_util, train_util, mesh_util, io_util
from models.frameworks.neussegm import cdf_Phi_s, alpha_to_w, sdf_to_w, sdf_to_alpha

import os
import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from zju_smpl.lbs import batch_rodrigues

class SingleRenderer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)

def map_mesh_feature_to_volume(xyz, voxel_size=[0.005, 0.005, 0.005], enlarge_box=True, params=None):
    if len(xyz.shape)<3:
        xyz = xyz[None]
    if len(params['R'].shape)<3:
        params['R'] = params['R'][None]
        params['T'] = params['T'][None]
    B = xyz.shape[0]

    # transform smpl from the world coordinate to the smpl coordinate
    # Rh = params['Rh'][:,0,:].to(xyz.device)
    # R = batch_rodrigues(Rh)
    # T = params['Th'][:, 0, :].to(xyz.device)
    R = params['R'].to(xyz.device)
    T = params['T'].to(xyz.device)

    xyz = (xyz - T)@R

    # obtain the bounds for coord construction
    min_xyz = torch.min(xyz, dim=1, keepdim=True).values
    max_xyz = torch.max(xyz, dim=1, keepdim=True).values
    if enlarge_box>0.:
        min_xyz -= 0.05
        max_xyz += 0.05
    else:
        min_xyz[..., 2] -= 0.05
        max_xyz[..., 2] += 0.05
    bounds = torch.cat([min_xyz, max_xyz], dim=1)

    # construct the coordinate
    dhw = xyz[..., [2, 1, 0]]
    min_dhw = min_xyz[..., [2, 1, 0]]
    max_dhw = max_xyz[..., [2, 1, 0]]
    voxel_size = torch.tensor(voxel_size).to(xyz.device)
    grid_verts = torch.round((dhw - min_dhw) / voxel_size).type(torch.int32)

    # construct the output shape
    volume_shape = torch.ceil((max_dhw - min_dhw) / voxel_size).type(torch.int32)
    x = 32
    # Make out_sh to be divided by 32 to sparseconv operates with downsampled voxels
    volume_shape = (volume_shape | (x - 1)) + 1
    volume_shape = volume_shape.max(dim=0).values[0]
    batch_idx = torch.arange(0, B).view(-1,B).repeat((grid_verts.shape[1], 1)).T.reshape(-1).to(grid_verts.device).type(torch.int32)
    grid_verts = torch.cat([batch_idx.view(-1,1), grid_verts.view(-1, 3)], dim=1)
    return grid_verts, volume_shape, bounds, R, T


def volume_render(
        rays_o,
        rays_d,
        model,
        vertices:torch.Tensor=None,
        obj_bounding_radius=1.0,

        batched=False,
        batched_info={},

        # render algorithm config
        use_view_dirs=True,
        rayschunk=65536,
        netchunk=1048576,
        white_bkgd=False,
        near_bypass: Optional[float] = None,
        far_bypass: Optional[float] = None,

        # render function config
        detailed_output=True,
        show_progress=False,

        # sampling related
        perturb=False,  # config whether you do stratified sampling
        N_samples=64,
        N_importance=64,
        N_outside=0,

        # upsample related

        # featuremap related
        bounding_box=None,

        # neuralbody specific
        voxel_size=[0.005, 0.005, 0.005],
        enlarge_box=0.05,
        smpl_param:dict() =None,
        frame_latent_ind:torch.Tensor=torch.tensor([0]),
        near=None,
        far=None,
        **dummy_kwargs  # just place holder
):
    """
    input:
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]
    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)

    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)
    grid_verts, volume_shape, bounds, R, Th = map_mesh_feature_to_volume(vertices, voxel_size=voxel_size, enlarge_box=enlarge_box, params=smpl_param)
    feature_volume = model.encode_sparse_voxel(grid_verts=grid_verts, volume_shape=volume_shape, batch_size=B)

    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor, near: torch.Tensor=None, far: torch.Tensor=None):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]

        # [(B), N_rays] x 2
        if (near is None) and (far is None):
            if bounding_box is None:
                near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
            else:
                near, far, valididx = rend_util.near_far_from_bbox(rays_o, rays_d, bounding_box, margin=enlarge_box+0.05)
                assert torch.all(valididx).item(), "Some rays are outside bounding box"

            if near_bypass is not None:
                near = near_bypass * torch.ones_like(near).to(device)
            if far_bypass is not None:
                far = far_bypass * torch.ones_like(far).to(device)

        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None

        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]

        # ---------------
        # Sample points on the rays
        # ---------------

        # ---------------
        # Coarse Points
        # [(B), N_rays, N_samples]
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = near * (1 - _t) + far * _t

        # NOTE. swheo: Original neuralbody has no importance sampling
        if perturb:
            mids = .5 * (d_coarse[..., 1:] + d_coarse[..., :-1])
            upper = torch.cat([mids, d_coarse[..., -1:]], -1)
            lower = torch.cat([d_coarse[..., :1], mids], -1)
            t_rand = torch.rand(upper.shape).float().to(device)
            d_final = lower + (upper - lower) * t_rand
        else:
            d_final = d_coarse

        # ------------------
        # Calculate Points
        # [(B), N_rays, N_samples+N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_final[..., :, None]

        sample_range = torch.stack([pts.reshape((-1, 3)).min(0)[0], pts.reshape((-1, 3)).max(0)[0]])[None, None, ...]

        # ------------------
        # Inside Scene
        # ------------------
        xyz_features, _ = batchify_query(model.sample_voxel_feature, pts, feature_volume=feature_volume,
                                         R=R, Th=Th, bounds=bounds, voxel_size=voxel_size,
                                         volume_shape=volume_shape)

        sigma, radiance, logit = batchify_query(model.forward, pts, view_dirs.unsqueeze(-2).expand_as(pts), xyz_features,
                                    latent_idx=frame_latent_ind, compute_rgb=True, compute_seg=True if model.segm_net is not None else False)

        # --------------
        # Ray Integration
        # --------------
        dists = d_final[..., 1:] - d_final[..., :-1]
        dists = torch.cat(
            [dists,
             torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)],
            -1)  # [N_rays, N_samples]

        # Convert density to alpha and compute visibility weights
        opacity_alpha = 1. - torch.exp(-F.relu(sigma) * dists)

        #
        visibility_weights = opacity_alpha * torch.cumprod(
            torch.cat(
                [torch.ones((*opacity_alpha.shape[:2], 1)).to(opacity_alpha), 1. - opacity_alpha + 1e-10],
                -1), -1)[..., :-1]
        rgb_map = torch.sum(visibility_weights[..., None] * radiance, -2)  # [N_rays, 3]

        depth_map = torch.sum(visibility_weights * d_final, -1)
        # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
        #                           depth_map / torch.sum(visibility_weights, -1))
        acc_map = torch.sum(visibility_weights, -1)

        # ------------------
        # Outside Scene
        # ------------------
        # NotImplemented

        # # NOTE: to get the correct depth map, the sum of weights must be 1!
        # # depth_map = torch.sum(visibility_weights / (visibility_weights.sum(-1, keepdim=True)+1e-10) * d_final, -1)
        if model.segm_net is not None:
            logit_map = torch.sum(visibility_weights[..., None] * logit, -2)
            # acc_map = torch.sum(visibility_weights, -1)
            label_map = torch.argmax(F.softmax(logit_map, -1), -1)
            label_map_color = model.labels_cmap[label_map]

        # mask_weights = alpha_to_w(alpha_in)
        # acc_map = torch.sum(mask_weights, -1)
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),  # [(B), N_rays, 3]
            ('depth_volume', depth_map),  # [(B), N_rays]
            # ('depth_surface', d_pred_out),    # [(B), N_rays]
            ('mask_volume', acc_map),  # [(B), N_rays]
        ])

        if model.segm_net is not None:
            ret_i['logit'] = logit_map # [(B), N_rays, N_classes]
            ret_i['label_map'] = label_map
            ret_i['label_map_color'] = label_map_color

        # if calc_normal:
        #     normals_map = F.normalize(nablas, dim=-1)
        #     N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
        #     normals_map = (normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)
        #     ret_i['normals_volume'] = normals_map

        if detailed_output:
            ret_i['density'] = sigma
            ret_i['radiance'] = radiance
            ret_i['alpha'] = opacity_alpha
            ret_i['visibility_weights'] = visibility_weights
            ret_i['d_final'] = d_final
            ret_i['sample_range'] = sample_range

        return ret_i

    ret = {}
    for i in tqdm(range(0, rays_d.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        if (near is None) and (far is None):
            ret_i = render_rayschunk(
                rays_o[:, i:i + rayschunk] if batched else rays_o[i:i + rayschunk],
                rays_d[:, i:i + rayschunk] if batched else rays_d[i:i + rayschunk],
            )
        else:
            ret_i = render_rayschunk(
                rays_o[:, i:i + rayschunk] if batched else rays_o[i:i + rayschunk],
                rays_d[:, i:i + rayschunk] if batched else rays_d[i:i + rayschunk],
                near[:, i:i + rayschunk] if batched else near[i:i + rayschunk],
                far[:, i:i + rayschunk] if batched else far[i:i + rayschunk],
            )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)

    return ret['rgb'], ret['depth_volume'], ret