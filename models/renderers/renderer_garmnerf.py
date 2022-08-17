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


class SingleRenderer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)

def volume_render(
        rays_o,
        rays_d,
        model,

        obj_bounding_radius=1.0,

        batched=False,
        batched_info={},

        # render algorithm config
        calc_normal=False,
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
        perturb=False,  # config whether do stratified sampling
        fixed_s_recp=1 / 64.,
        N_samples=64,
        N_importance=64,
        N_outside=0,

        # upsample related
        upsample_algo='official_solution',
        N_nograd_samples=2048,
        N_upsample_iters=4,

        # featuremap related
        cbfeat_map=None,
        idfeat_map=None,
        smpl_param=None,
        bounding_box=None,
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
    # todo smaller dim of rays_o for better memory usage
    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)

    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)
    cbfeat, idGfeat, idCfeat, idSfeat = model.forward_featext(cbuvmap=cbfeat_map, iduvmap=idfeat_map)

    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]

        # [(B), N_rays] x 2
        if bounding_box is None: #todo use this if near far are not provided given near far if
            near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
        else:
            near, far, valididx = rend_util.near_far_from_bbox(rays_o, rays_d, bounding_box)
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

        # ---------------
        # Up Sampling
        with torch.no_grad():
            # -------- option 1: directly use
            if upsample_algo == 'direct_use':  # nerf-like
                # [(B), N_rays, N_samples, 3]
                pts_coarse = rays_o.unsqueeze(-2) + d_coarse.unsqueeze(-1) * rays_d.unsqueeze(-2)
                # query network to get sdf
                # [(B), N_rays, N_samples]
                # sdf_coarse = model.implicit_surface.forward(pts_coarse)
                sdf_coarse, _ = model.forward_sdf(pts_coarse, cbfeat, idGfeat, idCfeat, idSfeat, smpl_param)
                # [(B), N_rays, N_samples-1]
                *_, w_coarse = sdf_to_w(sdf_coarse, 1. / fixed_s_recp)
                # Fine points
                # [(B), N_rays, N_importance]
                d_fine = rend_util.sample_pdf(d_coarse, w_coarse, N_importance, det=not perturb)
                # Gather points
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)

            # -------- option 2: just using more points to calculate visibility weights for upsampling
            # used config: N_nograd_samples
            elif upsample_algo == 'direct_more':
                _t = torch.linspace(0, 1, N_nograd_samples).float().to(device)
                _d = near * (1 - _t) + far * _t
                _pts = rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2)
                # _sdf = model.implicit_surface.forward(_pts)
                _sdf = batchify_query(model.forward_sdf, _pts, cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat,
                                      idSfeat=idSfeat, smpl_param=smpl_param)
                *_, _w = sdf_to_w(_sdf, 1. / fixed_s_recp)
                d_fine = rend_util.sample_pdf(_d, _w, N_importance, det=not perturb)
                # Gather points
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)

            # -------- option 3: modified from NeuS official implementation: estimate sdf slopes and middle points' sdf
            # https://github.com/Totoro97/NeuS/blob/9dc9275d3a8c7266994a3b9cf9f36071621987dd/models/renderer.py#L131
            # used config: N_upsample_iters
            elif upsample_algo == 'official_solution':
                _d = d_coarse
                _sdf = batchify_query(model.forward_sdf, rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2),
                                      cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat,
                                      smpl_param=smpl_param)[0]
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = _sdf[..., :-1], _sdf[..., 1:]
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
                    prev_dot_val = torch.cat([torch.zeros_like(dot_val[..., :1], device=device), dot_val[..., :-1]],
                                             dim=-1)  # jianfei: prev_slope, right shifted
                    dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)  # jianfei: concat prev_slope with slope
                    dot_val, _ = torch.min(dot_val, dim=-1,
                                           keepdim=False)  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = (next_z_vals - prev_z_vals)
                    prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                    prev_cdf = cdf_Phi_s(prev_esti_sdf, 64 * (2 ** i))
                    next_cdf = cdf_Phi_s(next_esti_sdf, 64 * (2 ** i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                    _w = alpha_to_w(alpha)
                    d_fine = rend_util.sample_pdf(_d, _w, N_importance // N_upsample_iters, det=not perturb)
                    _d = torch.cat([_d, d_fine], dim=-1)

                    sdf_fine = batchify_query(model.forward_sdf,
                                              rays_o.unsqueeze(-2) + d_fine.unsqueeze(-1) * rays_d.unsqueeze(-2),
                                              cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat,
                                              smpl_param=smpl_param)[0]
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, DIM_BATCHIFY + 1, d_sort_indices)
                d_all = _d
            else:
                raise NotImplementedError

        # ------------------
        # Calculate Points
        # [(B), N_rays, N_samples+N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        # [(B), N_rays, N_pts-1, 3]
        # pts_mid = 0.5 * (pts[..., 1:, :] + pts[..., :-1, :])
        d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
        pts_mid = rays_o[..., None, :] + rays_d[..., None, :] * d_mid[..., :, None]
        sample_range = torch.stack([pts.reshape((-1, 3)).min(0)[0], pts.reshape((-1, 3)).max(0)[0]])[None, None, ...]
        #todo, shift mask and take one other points outside mask than use it for volume rendering?
        #

        # if bounding_box is not None:
        #     inside_mask = torch.all(pts>=bounding_box[None,None,0,:], dim=-1) &\
        #                   torch.all(pts<=bounding_box[None,None,1,:], dim=-1)
        #     inside_mask_mid = torch.all(pts_mid>=bounding_box[None,None,0,:], dim=-1) & \
        #                       torch.all(pts_mid<=bounding_box[None,None,1,:], dim=-1)
        # else:
        #     inside_mask = torch.ones_like(pts[...,0]) > 0.
        #     inside_mask_mid = inside_mask = torch.ones_like(pts_mid[...,0]) > 0.
        # inside_mask = torch.ones_like(pts[..., 0]) > 0.
        # inside_mask_mid = inside_mask = torch.ones_like(pts_mid[..., 0]) > 0.

        # ------------------
        # Inside Scene
        # ------------------
        sdf, nablas, _, _ = batchify_query(model.forward_sdf_with_nablas, pts, cbfeat=cbfeat, idGfeat=idGfeat,
                                           idCfeat=idCfeat, idSfeat=idSfeat,
                                           smpl_param=smpl_param)#, inside_mask=inside_mask)
        # sdf[~inside_mask] = obj_bounding_radius
        # [(B), N_ryas, N_pts], [(B), N_ryas, N_pts-1]
        cdf, opacity_alpha = sdf_to_alpha(sdf, model.decoder.forward_s())
        # radiances = model.forward_radiance(pts_mid, view_dirs_mid)
        radiances = batchify_query(model.forward_radiance, pts_mid,
                                   view_dirs.unsqueeze(-2).expand_as(pts_mid) if use_view_dirs else None,
                                   cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat, smpl_param=smpl_param)
        if model.decoder.segm_net is not None:
            logits = batchify_query(model.forward_segm, pts_mid,
                                    view_dirs.unsqueeze(-2).expand_as(pts_mid) if use_view_dirs else None,
                                    cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat, smpl_param=smpl_param)
        # ------------------
        # Outside Scene
        # ------------------
        if (N_outside > 0) and (bounding_box is None):
            _t = torch.linspace(0, 1, N_outside + 2)[..., 1:-1].float().to(device)
            d_vals_out = far / torch.flip(_t, dims=[-1])  # sorting by flip 1/_t (far ~ 1/min(_t))
            if perturb:
                _mids = .5 * (d_vals_out[..., 1:] + d_vals_out[..., :-1])
                _upper = torch.cat([_mids, d_vals_out[..., -1:]], -1)
                _lower = torch.cat([d_vals_out[..., :1], _mids], -1)
                _t_rand = torch.rand(_upper.shape).float().to(device)
                d_vals_out = _lower + (_upper - _lower) * _t_rand

            d_vals_out = torch.cat([d_mid, d_vals_out], dim=-1)  # already sorted
            pts_out = rays_o[..., None, :] + rays_d[..., None, :] * d_vals_out[..., :, None]
            r = pts_out.norm(dim=-1, keepdim=True)
            x_out = torch.cat([pts_out / r, 1. / r], dim=-1)
            views_out = view_dirs.unsqueeze(-2).expand_as(x_out[..., :3]) if use_view_dirs else None

            sigma_out, radiance_out, logits_out = batchify_query(model.decoder.nerf_outside.forward, x_out, views_out)
            dists = d_vals_out[..., 1:] - d_vals_out[..., :-1]  # step
            dists = torch.cat([dists, 1e10 * torch.ones(dists[..., :1].shape).to(device)], dim=-1)
            alpha_out = 1 - torch.exp(
                -F.softplus(sigma_out) * dists)  # use softplus instead of relu as NeuS's official repo

            # --------------
            # Ray Integration
            # --------------
            # [(B), N_rays, N_pts-1]
            N_pts_1 = d_mid.shape[-1]
            # [(B), N_ryas, N_pts-1]
            if True: #bounding_box is None: #todo
                mask_inside = (pts_mid.norm(dim=-1) <= obj_bounding_radius)
            else:
                mask_inside = torch.all(torch.logical_and(pts_mid >= bounding_box[0], pts_mid <= bounding_box[1:2]),
                                        dim=-1)
            # [(B), N_ryas, N_pts-1]
            alpha_in = opacity_alpha * mask_inside.float() + alpha_out[..., :N_pts_1] * (~mask_inside).float()
            # [(B), N_ryas, N_pts-1 + N_outside]
            opacity_alpha = torch.cat([alpha_in, alpha_out[..., N_pts_1:]], dim=-1)

            # [(B), N_ryas, N_pts-1,+ 3]
            radiance_in = radiances * mask_inside.float()[..., None] + radiance_out[..., :N_pts_1, :] * \
                          (~mask_inside).float()[..., None]
            # [(B), N_ryas, N_pts-1 + N_outside, 3]
            radiances = torch.cat([radiance_in, radiance_out[..., N_pts_1:, :]], dim=-2)

            if model.decoder.segm_net is not None:
                logits_in = logits * mask_inside.float()[..., None] + logits_out[..., :N_pts_1, :] * (~mask_inside).float()[
                    ..., None]
                logits = torch.cat([logits_in, logits_out[..., N_pts_1:, :]], dim=-2)
            d_final = d_vals_out
        else:
            alpha_in = opacity_alpha
            d_final = d_mid
        # --------------
        # Ray Integration
        # --------------
        # # [(B), N_rays, N_pts-1]
        # d_final = d_mid

        # [(B), N_ryas, N_pts-1]
        visibility_weights = alpha_to_w(opacity_alpha)
        # [(B), N_rays]
        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)
        depth_map = torch.sum(visibility_weights * d_final, -1)
        # NOTE: to get the correct depth map, the sum of weights must be 1!
        # depth_map = torch.sum(visibility_weights / (visibility_weights.sum(-1, keepdim=True)+1e-10) * d_final, -1)
        if model.decoder.segm_net is not None:
            logit_map = torch.sum(visibility_weights[..., None] * logits, -2)
            # acc_map = torch.sum(visibility_weights, -1)
            label_map = torch.argmax(F.softmax(logit_map, -1), -1)
            label_map_color = model.decoder.labels_cmap[label_map]


        mask_weights = alpha_to_w(alpha_in)
        acc_map = torch.sum(mask_weights, -1)
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),  # [(B), N_rays, 3]
            ('depth_volume', depth_map),  # [(B), N_rays]
            # ('depth_surface', d_pred_out),    # [(B), N_rays]
            ('mask_volume', acc_map),  # [(B), N_rays]
        ])

        if model.decoder.segm_net is not None:
            ret_i['logit'] = logit_map # [(B), N_rays, N_classes]

        if calc_normal:
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
            normals_map = (normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)
            ret_i['normals_volume'] = normals_map

        if detailed_output:
            ret_i['implicit_nablas'] = nablas
            ret_i['implicit_surface'] = sdf
            ret_i['radiance'] = radiances
            ret_i['alpha'] = opacity_alpha
            ret_i['cdf'] = cdf
            ret_i['visibility_weights'] = visibility_weights
            ret_i['d_final'] = d_final
            ret_i['sample_range'] = sample_range
            if model.decoder.segm_net is not None:
                ret_i['label_map'] = label_map
                ret_i['label_map_color'] = label_map_color

        return ret_i

    ret = {}
    for i in tqdm(range(0, rays_d.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i + rayschunk] if batched else rays_o[i:i + rayschunk],
            rays_d[:, i:i + rayschunk] if batched else rays_d[i:i + rayschunk]
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)

    return ret['rgb'], ret['depth_volume'], ret