from models.base import NerfppNetwithAutoExpo
from utils import rend_util, train_util

import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def perturb_samples(z_vals):
    # From nerfpp code
    _mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    _upper = torch.cat([_mids, z_vals[..., -1:]], -1)
    _lower = torch.cat([z_vals[..., :1], _mids], -1)
    # uniform samples in those intervals
    _t_rand = torch.rand(_upper.shape).float().to(z_vals.device)
    z_vals = _lower + (_upper - _lower) * _t_rand

    return z_vals

def render_nerfpp(rays_o, rays_d, d_fg_max, d_fg, d_bg, use_view_dirs, model, indices, batchify_query=None):
    pts_fg = rays_o[..., None, :] + rays_d[..., None, :] * d_fg[..., :, None]
    x_fg = pts_fg
    pts_bg = rays_o[..., None, :] + rays_d[..., None, :] * d_bg[..., :, None]
    r = pts_bg.norm(dim=-1, keepdim=True)
    x_bg = torch.cat([pts_bg / r, 1. / r], dim=-1)

    if use_view_dirs:
        view_dirs = rays_d
    else:
        view_dirs = None

    views_in = view_dirs.unsqueeze(-2).expand_as(x_bg[..., :3]) if use_view_dirs else None

    # (sigma, radiance)
    if batchify_query is not None:
        if indices==-1:
            fg_sigma, fg_radiance, bg_sigma, bg_radiance, scale, shift = batchify_query(model.forward, x_fg, x_bg, views_in)
        else:
            indices_ = indices.expand_as(d_fg).unsqueeze(-1)
            fg_sigma, fg_radiance, bg_sigma, bg_radiance, scale, shift = batchify_query(model.forward, x_fg, x_bg, views_in, indices_)
    else:
        fg_sigma, fg_radiance, bg_sigma, bg_radiance, scale, shift = model(x_fg, x_bg, views_in, indices)

    # Need to do this because of batchify func.
    out_expo = (scale[...,0], shift[...,0])

    # Render foreground
    fg_dists = d_fg[..., 1:] - d_fg[..., :-1]  # step
    fg_dists = torch.cat([fg_dists, d_fg_max - d_fg[..., -1:]], dim=-1)
    fg_alpha = 1. - torch.exp(-F.softplus(fg_sigma) * fg_dists)
    opacity = torch.cumprod(1. - fg_alpha + 1e-10, dim=-1)  # todo compare to volsdf one
    bg_lambda = opacity[..., -1]  # todo check clone?
    opacity = torch.cat((torch.ones_like(opacity[..., 0:1]), opacity[..., :-1]), dim=-1)
    fg_weights = fg_alpha * opacity
    fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_radiance, dim=-2)
    fg_depth_map = torch.sum(fg_weights * d_fg, dim=-1)
    fg_acc_map = torch.sum(fg_weights, dim=-1)
    # fg_weights/(fg_weights.sum(-1, keepdim=True)+1e-10)
    # Render background
    bg_dists = d_bg[..., 1:] - d_bg[..., :-1]
    bg_dists = torch.cat((bg_dists, 1e10 * torch.ones_like(bg_dists[..., 0:1])), dim=-1)
    bg_alpha = 1. - torch.exp(-F.softplus(bg_sigma) * bg_dists)
    opacity = torch.cumprod(1. - bg_alpha + 1e-10, dim=-1)[..., :-1]
    opacity = torch.cat((torch.ones_like(opacity[..., 0:1]), opacity), dim=-1)
    bg_weights = bg_alpha * opacity
    bg_rgb_map = bg_lambda.unsqueeze(-1) * torch.sum(bg_weights.unsqueeze(-1) * bg_radiance, dim=-2)
    bg_depth_map = bg_lambda * torch.sum(bg_weights * d_bg, dim=-1)
    bg_acc_map = bg_lambda * torch.sum(bg_weights, dim=-1)

    rgb_map = fg_rgb_map + bg_rgb_map
    depth_map = fg_depth_map + bg_depth_map
    acc_map = fg_acc_map + bg_acc_map

    ret = OrderedDict([('rgb', rgb_map),
                       ('depth', depth_map),
                       ('mask', acc_map),
                       ('autoexpo', out_expo),
                       ('fg_weights', fg_weights),
                       ('bg_weights', bg_weights)
                       ])

    return ret

class NeRFpp(nn.Module):
    def __init__(self,
                 input_ch=3,
                 num_images=-1,
                 coarse_cfg = dict(),
                 fine_cfg = dict(),
                 ):
        super().__init__()

        """
        models['coarsenet']:NerfppNetwithAutoExpo
        models['finenet']:NerfppNetwithAutoExpo
        """
        self.coarsenet = NerfppNetwithAutoExpo(input_ch = input_ch, num_images=num_images, **coarse_cfg)
        self.finenet   = NerfppNetwithAutoExpo(input_ch = input_ch, num_images=num_images, **fine_cfg)

    def foward(self, x_fg: torch.Tensor, x_bg: torch.Tensor, view_dirs, indices=-1, level='finenet'):
        return self.coarsenet(x_fg, x_bg, view_dirs, indices) if level=='coarsent'\
            else self.finenet(x_fg, x_bg, view_dirs, indices)

class SingleRenderer(nn.Module):
    def __init__(self, model: OrderedDict):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


def volume_render(
        rays_o,
        rays_d,
        model,
        # Autoexpo sepcific
        indices = torch.Tensor([-1]).type(torch.IntTensor),
        obj_bounding_radius=1.0,  # UnitSphere

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
        perturb=False,  # config whether do stratified sampling
        N_coarse=64,  # 64/64 inside/outside (code), total 128
        N_importance=64,  # 128/128 inside/outside (paper), total 256

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

    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]
        prefix_batch = [B] if batched else []

        # [(B), N_rays] x 2
        near, _ = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
        # nears = near * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        _, far, mask_intersect = rend_util.get_sphere_intersection(rays_o, rays_d, r=obj_bounding_radius)
        assert mask_intersect.all()

        # ---------------
        # Cascade ('coarse, fine')
        # ---------------
        levels = ['coarsenet', 'finenet']
        # todo flexible cascade?
        ret_i = OrderedDict()
        for k in levels:
            if k is 'coarsenet':
                # ---------------
                # Coarse Points
                # [(B), N_rays, N_samples]
                ##todo : Original nerfpp's sample computation (rotation) vs NeuS's computation
                _t = torch.linspace(0, 1, N_coarse).float().to(device)
                d_fg = near * (1 - _t) + far * _t
                if perturb:
                    d_fg = perturb_samples(d_fg)
                _t = torch.linspace(0, 1, N_coarse + 2)[..., 1:-1].float().to(device)
                if perturb:
                    _t = perturb_samples(_t)
                d_bg = far / torch.flip(_t, dims=[-1])  # sorting by flip 1/_t (far ~ 1/min(_t))
            else:
                # ---------------
                # Fine sampling from pdf and concat with earlier samples
                # [(B), N_rays, N_samples]
                # fg_weights = ret_level['fg_weights'].clone().detach()[..., 1:-1]
                # If detach, it will detach nerf's layers so thus the resulting gradients are none
                fg_weights = ret_level['fg_weights'].clone()[..., 1:-1]
                d_fg_mid = .5 * (d_fg[..., 1:] + d_fg[..., :-1])
                # Note. weights are normalized inside the sample_pdf. No worry
                d_fg_samples = rend_util.sample_pdf(bins=d_fg_mid, weights=fg_weights,
                                          N_importance=N_importance, det=False)
                d_fg, _ = torch.sort(torch.cat((d_fg, d_fg_samples), dim=-1))

                # bg_weights = ret_level['bg_weights'].clone().detach()[..., 1:-1]
                bg_weights = ret_level['bg_weights'].clone()[..., 1:-1]
                d_bg_mid = .5 * (d_bg[..., 1:] + d_bg[..., :-1])
                # Note. weights are normalized inside the sample_pdf. No worry
                d_bg_samples = rend_util.sample_pdf(bins=d_bg_mid, weights=bg_weights,
                                          N_importance=N_importance, det=False)
                d_bg, _ = torch.sort(torch.cat((d_bg, d_bg_samples), dim=-1))

            ret_level = render_nerfpp(rays_o, rays_d, far, d_fg, d_bg, use_view_dirs, getattr(model, k),
                                      indices, batchify_query=batchify_query)
            # ret_level = OrderedDict([('rgb', rgb_map),
            #        ('depth', depth_map),
            #        ('autoexpo', out_expo),
            #        ('fg_weights', fg_weights),
            #        ('bg_weights', bg_weights)
            #       ])
            rgb_map = ret_level['rgb']
            scale, shift = ret_level['autoexpo']
            # todo scalar log of scale and shift?
            rgb_map = (rgb_map - shift[...,0]) / scale[...,0]

            ret_i[f'{k}/autoexpo_scale'] = scale
            ret_i[f'{k}/autoexpo_shift'] = shift

            ret_i[f'{k}/rgb'] = rgb_map  # [(B), N_rays, 3]
            ret_i[f'{k}/depth_volume'] = ret_level['depth']  # [(B), N_rays]
            ret_i[f'{k}/mask_volume'] = ret_level['mask']  # [(B), N_rays]



            if detailed_output:
                pass
                # todo: tbd, for detailed output?

        return ret_i


    ret = {}
    for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
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

    ret['mask_volume'] = ret['finenet/mask_volume']
    ret['normals_volume'] = torch.zeros_like(ret['finenet/rgb'])
    return ret['finenet/rgb'], ret['finenet/depth_volume'], ret


class Trainer(nn.Module):
    def __init__(self, model: dict, device_ids=[0], batched=True):
        super().__init__()
        self.model = model
        self.renderer = SingleRenderer(model)
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]

    def forward(self,
                args,
                indices,
                model_input,
                ground_truth,
                render_kwargs_train: dict,
                it: int,
                device='cuda'):

        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, render_kwargs_train['H'], render_kwargs_train['W'], N_rays=args.data.N_rays,
            jittered=args.training.jittered if hasattr(args.training, 'jittered') else False)
        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3 * [select_inds], -1))

        if "mask_ignore" in model_input:
            mask_ignore = torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
        else:
            mask_ignore = None

        rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)

        # [B, N_rays, N_pts, 3]
        # nablas: torch.Tensor = extras['implicit_nablas']
        # [B, N_rays, N_pts]
        # nablas_norm = torch.norm(nablas, dim=-1)
        # [B, N_rays]
        mask_volume: torch.Tensor = extras['mask_volume']
        # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
        # swheo, Note: dueto binary cross entropy (log(0) = -inf) - exploding
        # mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
        mask_volume = torch.clamp(mask_volume, 1e-3, 1 - 1e-3)
        extras['mask_volume_clipped'] = mask_volume

        losses = OrderedDict()

        fine_rgb = extras['finenet/rgb']
        coarse_rgb = extras['coarsenet/rgb']
        # [B, N_rays, 3]
        # todo : L2 loss in nerfplusplus
        # losses['loss_img'] = F.l1_loss(fine_rgb, target_rgb, reduction='none') + F.l1_loss(coarse_rgb, target_rgb, reduction='none')
        losses['loss_img'] = F.mse_loss(fine_rgb, target_rgb, reduction='none') + F.mse_loss(coarse_rgb, target_rgb,
                                                                                           reduction='none')

        # [B, N_rays, N_pts]
        # losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')

        # use_autoexpo = True if 'finenet/autoexpo_scale' in extras else False
        if args.training.use_autoexpo:
            scale = extras['finenet/autoexpo_scale']
            shift = extras['finenet/autoexpo_shift']

            losses['loss-autoexpo'] = args.training.w_autoexpo * (torch.abs(scale - 1) + torch.abs(shift)).mean()

        if args.training.with_mask:
            # [B, N_rays]
            target_mask = torch.gather(model_input["object_mask"].to(device), 1, select_inds)
            losses['loss_mask'] = args.training.w_mask * F.binary_cross_entropy(mask_volume, target_mask.float(),
                                                                                reduction='mean')
            if mask_ignore is not None:
                target_mask = torch.logical_and(target_mask, mask_ignore)
            # [N_masked, 3]
            losses['loss_img'] = (losses['loss_img'] * target_mask[..., None].float()).sum() / (
                        target_mask.sum() + 1e-10)
        else:
            if mask_ignore is not None:
                losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (
                            mask_ignore.sum() + 1e-10)
            else:
                losses['loss_img'] = losses['loss_img'].mean()

        loss = 0
        for k, v in losses.items():
            loss += losses[k]

        losses['total'] = loss
        extras['scalars'] = {} #{'finenet/autoexpo_scale': self.model['finenet'].} #todo mean scale, shift for whole data
        extras['select_inds'] = select_inds

        # sample_range = extras.pop('sample_range').reshape((-1, 2, 3))
        # range_min = sample_range[:, 0, :].min(0)[0]
        # range_max = sample_range[:, 1, :].max(0)[0]
        # extras['scalars'].update(
        #     {'minx': range_min[0], 'miny': range_min[1], 'minz': range_min[2]})
        # extras['scalars'].update(
        #     {'maxx': range_max[0], 'maxy': range_max[1], 'maxz': range_max[2]})

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])


def get_model(args):
    model_cfg = {
    }
    coarse_cfg = {
        'embed_multires': args.model.coarse.setdefault('embed_multires', 6),
        'use_view_dirs': args.model.coarse.setdefault('use_view_dirs', True),
        'embed_multires_view': args.model.coarse.setdefault('embed_multires_view', -1),
        'D': args.model.coarse.setdefault('D', 8),
        'W': args.model.coarse.setdefault('W', 256),
        'skips': args.model.coarse.setdefault('skips', [4]),
    }
    fine_cfg = {
        'embed_multires': args.model.fine.setdefault('embed_multires', 6),
        'use_view_dirs': args.model.fine.setdefault('use_view_dirs', True),
        'embed_multires_view': args.model.fine.setdefault('embed_multires_view', -1),
        'D': args.model.fine.setdefault('D', 8),
        'W': args.model.fine.setdefault('W', 256),
        'skips': args.model.fine.setdefault('skips', [4]),
    }
    model_cfg['coarse_cfg'] = coarse_cfg
    model_cfg['fine_cfg'] = fine_cfg
    model = NeRFpp(**model_cfg)

    ## render kwargs
    render_kwargs_train = {
        'N_coarse': args.model.setdefault('N_coarse', 64),
        'N_importance': args.model.setdefault('N_coarse', 64),

        'obj_bounding_radius': args.model.setdefault('obj_bounding_radius', 1.0),
        'batched': args.data.batch_size is not None,
        'perturb': args.model.setdefault('perturb', True),  # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False

    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])

    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer
