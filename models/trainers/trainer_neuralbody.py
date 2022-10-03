from utils import rend_util, train_util, mesh_util, io_util
from models.renderers.renderer_neuralbody import SingleRenderer, map_mesh_feature_to_volume

import os
import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from kornia.morphology import dilation

import matplotlib.cm as colormap


class Trainer(nn.Module):
    def __init__(self, model, device_ids=[0], batched=True):
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
                detailed_output=True,
                device='cuda'):
        B, H, W = model_input['object_mask'].shape
        # mask = model_input['object_mask'].reshape(B, -1).to(device)
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        smplparams = model_input['smpl_params']
        vertices = model_input['vertices'].to(device)
        # NB specific
        frame_latent_ind = model_input['frame_latent_ind'].to(device)
        # Computed object bounding bbox
        # bbox = torch.stack([vertices.view(-1,3).min(dim=0).values, vertices.view(-1,3).max(dim=0).values]).to(device)
        bbox = torch.stack([vertices.min(dim=1).values, vertices.max(dim=1).values], dim=1).to(device) # [B, 2, 3]
        # Just as neural body concept
        if render_kwargs_train['enlarge_box'] > 0.:
            bbox[:, 0, :] = bbox[:, 0, :] - render_kwargs_train['enlarge_box'] # swheo: for me, it was 0.2
            bbox[:, 1, :] = bbox[:, 1, :] + render_kwargs_train['enlarge_box']
        else:
            bbox[:, 0, :] = bbox[:, 0, :] - render_kwargs_train['enlarge_box']
            bbox[:, 1, :] = bbox[:, 1, :] + render_kwargs_train['enlarge_box']

        bounding_box = rend_util.get_2dmask_from_bbox(bbox, intrinsics, c2w, H, W)
        # mask = model_input['object_mask'].to(device)
        # bounding_box = model_input['bbox_mask'].to(device)
        # near, far, valididx = rend_util.near_far_from_bbox(rays_o, rays_d, bounding_box)
        # self.model.mesh.update_vertices(vertices)
        rays_o, rays_d, select_inds, near, far = \
            rend_util.get_rays_nb(c2w, intrinsics, H, W, bbox, args.data.N_rays,
                jittered=args.training.jittered if args.training.get('jittered') is not None else False,
                mask=bounding_box.reshape([-1, H, W])[0] if args.training.get('sample_maskonly', False) is not False else None)

        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3 * [select_inds], -1))
        target_segm = torch.gather(ground_truth['segm'].to(device), 1, select_inds)

        if "mask_ignore" in model_input:
            mask_ignore = torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
        else:
            mask_ignore = None
        # For debug
        """
        dd = bounding_box.reshape([-1, H, W])[0].detach().cpu().numpy()
        import matplotlib.pyplot as plt
        plt.imshow(dd)
        world_coords = rays_o + rays_d
        dddd = torch.bmm(torch.linalg.inv(c2w), torch.cat([world_coords, torch.ones_like(rays_o[...,0][...,None])], dim=-1).transpose(-1, -2)).transpose(-1,-2)
        vvv = torch.bmm(intrinsics, dddd.transpose(-1,-2))
        plt.scatter(vvv[0,0,:].detach().cpu().numpy(), vvv[0,1,:].detach().cpu().numpy())
        """
        rgb, depth_v, extras = self.renderer(rays_o, rays_d, vertices=vertices, detailed_output=detailed_output,
                                             smpl_param=smplparams,
                                             bounding_box=bbox.to(device)
                                             if args.training.get('sample_maskonly', False) is not False else None,
                                             frame_latent_ind=frame_latent_ind,
                                             near=near,
                                             far=far,
                                             **render_kwargs_train)

        # [B, N_rays]
        # mask_volume: torch.Tensor = extras['mask_volume']
        # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
        # Note. swheo, : dueto binary cross entropy (log(0) = -inf) - exploding
        # mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
        # mask_volume = torch.clamp(mask_volume, 1e-3, 1 - 1e-3)
        # extras['mask_volume_clipped'] = mask_volume

        losses = OrderedDict()

        # [B, N_rays, 3]
        losses['loss_img'] = F.mse_loss(rgb, target_rgb, reduction='mean')

        if 'logit' in extras:
            # [B, N_rays,]
            losses['loss_seg'] = args.training.w_seg * F.cross_entropy(extras['logit'].view((-1, 4)),
                                                                       target_segm.view((-1,)), reduction='mean')
        # if args.training.with_mask:
        #     # [B, N_rays]
        #     target_mask = torch.gather(mask.to(device), 1, select_inds)
        #     losses['loss_mask'] = args.training.w_mask * F.binary_cross_entropy(mask_volume, target_mask.float(),
        #                                                                         reduction='mean')
        #     if mask_ignore is not None:
        #         target_mask = torch.logical_and(target_mask, mask_ignore)
        #     # [N_masked, 3]
        #     losses['loss_img'] = (losses['loss_img'] * target_mask[..., None].float()).sum() / (
        #                 target_mask.sum() + 1e-10)
        #     # losses['loss_seg'] = (losses['loss_seg'] * target_mask[0].float()).sum() / (target_mask.sum()+1e-10)
        # else:
        #     if mask_ignore is not None:
        #         losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (
        #                 mask_ignore.sum() + 1e-10)
        #     else:
        #         losses['loss_img'] = losses['loss_img'].mean()

        loss = 0
        for k, v in losses.items():
            loss += losses[k]

        losses['total'] = loss
        extras['scalars'] = {}
        extras['select_inds'] = select_inds

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])

    def val(self,
            args,
            val_in,
            val_gt,
            it,
            render_kwargs_test: dict,
            device='cuda'):
        B, H, W = val_in['object_mask'].shape
        intrinsics = val_in["intrinsics"].to(device)
        c2w = val_in['c2w'].to(device)
        smplparams = val_in['smpl_params']
        vertices = val_in['vertices'].to(device)
        # self.model.mesh.update_vertices(vertices)
        # NB specific
        frame_latent_ind = val_in['frame_latent_ind'].to(device)
        mask_ = dilation(val_in['object_mask'].unsqueeze(0).to(device), torch.ones(11,11).to(device),
                         border_type='constant', border_value=0.)
        mask = mask_.reshape(B, -1) > 0.
        # Computed object bounding bbox
        # bbox = torch.stack([vertices.view(-1,3).min(dim=0).values, vertices.view(-1,3).max(dim=0).values]).to(device)
        bbox = torch.stack([vertices.min(dim=1).values, vertices.max(dim=1).values], dim=1).to(device) # [B, 2, 3]
        # Just as neural body concept
        if render_kwargs_test['enlarge_box'] > 0.:
            bbox[:, 0, :] = bbox[:, 0, :] - render_kwargs_test['enlarge_box'] # swheo: for me, it was 0.2
            bbox[:, 1, :] = bbox[:, 1, :] + render_kwargs_test['enlarge_box']
        else:
            bbox[:, 0, :] = bbox[:, 0, :] - render_kwargs_test['enlarge_box']
            bbox[:, 1, :] = bbox[:, 1, :] + render_kwargs_test['enlarge_box']

        if not render_kwargs_test['maskonly']:
            mask = None

        # N_rays=-1 for rendering full image,
        rays_o, rays_d, select_inds, near, far = \
            rend_util.get_rays_nb(c2w, intrinsics, H, W, bbox, -1, mask=mask)

        target_rgb = val_gt['rgb'].to(device)
        target_segm = val_gt['segm'].to(device)

        rgb, depth_v, ret = self.renderer(rays_o, rays_d, vertices=vertices, detailed_output=False,
                                          calc_normal=True,
                                          smpl_param=smplparams,
                                          bounding_box=bbox.to(device) if args.training.get('sample_maskonly',False) is not False else None,
                                          frame_latent_ind=frame_latent_ind,
                                          **render_kwargs_test)

        to_img = functools.partial(
            rend_util.lin2img,
            H=H, W=W,
            batched=render_kwargs_test['batched'])
        val_imgs = dict()
        val_imgs['val/gt_rgb'] = to_img(target_rgb[mask].unsqueeze(0), mask=mask) if mask is not None else to_img(target_rgb)
        if 'label_map_color' in ret:
            val_imgs['val/gt_segm'] = to_img(self.model.decoder.labels_cmap[target_segm])
        val_imgs['val/predicted_rgb'] = to_img(rgb, mask=mask)
        val_imgs['scalar/psnr'] = torchmetrics.functional.peak_signal_noise_ratio(val_imgs['val/gt_rgb'],
                                                                                  val_imgs['val/predicted_rgb'], data_range=1.0)
        val_imgs['scalar/ssim'] = torchmetrics.functional.structural_similarity_index_measure(val_imgs['val/gt_rgb'], val_imgs['val/predicted_rgb'], data_range=1.0)
        val_imgs['scalar/mse'] = torchmetrics.functional.mean_squared_error(val_imgs['val/gt_rgb'], val_imgs['val/predicted_rgb'])
        val_imgs['val/pred_depth_volume'] = to_img((depth_v / (depth_v.max() + 1e-10)).unsqueeze(-1), mask=mask)
        val_imgs['val/pred_mask_volume'] = to_img(ret['mask_volume'].unsqueeze(-1), mask=mask)
        # val_imgs['scalar/mask_iou'] = torchmetrics.functional.jaccard_index(???, val_imgs['val/pred_mask_volume'], num_classes=???)
        if 'depth_surface' in ret:
            val_imgs['val/pred_depth_surface'] = to_img(
                (ret['depth_surface'] / ret['depth_surface'].max()).unsqueeze(-1), mask=mask)
        if 'mask_surface' in ret:
            val_imgs['val/predicted_mask'] = to_img(ret['mask_surface'].unsqueeze(-1).float(), mask=mask)
        if 'label_map_color' in ret:
            val_imgs['val/predicted_segm'] = to_img(ret['label_map_color'], mask=mask)
            label_map_pred = to_img(ret['label_map'].unsqueeze(-1), mask=mask)
            label_map_gt = to_img(target_segm.unsqueeze(-1))
            val_imgs['scalar/mask_iou'] = torchmetrics.functional.jaccard_index(label_map_pred, label_map_gt, num_classes=self.model.decoder.labels_cmap.shape[0])

        return val_imgs

    def val_mesh(self,
                 args,
                 val_in,
                 filepath,
                 render_kwargs_test: dict,
                 device='cuda'
                 ):
        smplparams = val_in['smpl_params']
        vertices = val_in['vertices'].to(device)
        self.model.mesh.update_vertices(vertices)
        grid_verts, volume_shape, bounds, R, Th = map_mesh_feature_to_volume(self.model.mesh.vertices,
                                                                               voxel_size=render_kwargs_test['voxel_size'],
                                                                               enlarge_box=render_kwargs_test['enlarge_box'], params=smplparams)
        feature_volume = self.model.encode_sparse_voxel(grid_verts=grid_verts, volume_shape=volume_shape, batch_size=1)

        class fooFunc:
            def __init__(self, model):
                self.feature_sampler = functools.partial(model.sample_voxel_feature,
                                                         feature_volume=feature_volume, R=R, Th=Th,
                                                         bounds=bounds, voxel_size=render_kwargs_test['voxel_size'],
                                                         volume_shape=volume_shape)
                self.surface_func = model.density_net

            def forward(self, x):
                xyz_features, _ = self.feature_sampler(x)
                return self.surface_func(xyz_features)[0,...]

        # todo enable bbox based acceleration (just as neural body)
        implicit_surface = fooFunc(self.model)
        mesh_util.extract_mesh(
            implicit_surface,
            filepath=filepath,
            volume_size=args.data.get('volume_size', 2.0),
            chunk=16*1024,
            level=render_kwargs_test['mesh_th'],
            show_progress=True)

    def pretrain(
            self,
            args,
            dataloader,
            render_kwargs_test: dict,
            device='cuda',
            it=1, num_iters=5000, lr=1.0e-4, batch_points=6890,
            logger=None,
            **kwargs, #reserved
    ):
        raise NotImplementedError
