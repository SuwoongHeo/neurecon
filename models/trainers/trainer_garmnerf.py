from utils import rend_util, train_util, mesh_util, io_util
from models.renderers.renderer_garmnerf import SingleRenderer

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

def get_cos_anneal_ratio(it, anneal_end):
    if anneal_end == 0.0:
        return 1.0
    else:
        return np.min([1.0, it/anneal_end])

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
        mask = model_input['object_mask'].reshape(B, -1).to(device)
        # H, W = model_input['H'].item(), model_input['W'].item()
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        cbuvmap = model_input['cbuvmap'].to(device)
        idfeatmap = render_kwargs_train['idfeat_map_all'][model_input['subject_id'].item()][None].to(device) \
            if render_kwargs_train['idfeat_map_all'] != None else None
        smplparams = model_input['smpl_params']['poses'][:, 0, 3:].to(device) if args.data.smpl_feat != 'none' else None
        vertices = model_input['vertices'].to(device)
        # Computed object bounding bbox
        # margin = 0.2
        # bbox = torch.stack([vertices.min(dim=0).values-margin, vertices.max(dim=0).values+margin]).to(device)
        # bounding_box = rend_util.get_2dmask_from_bbox(bbox, intrinsics[0], c2w[0], H, W)[0]

        bbox = torch.stack([vertices.min(dim=1).values, vertices.max(dim=1).values], dim=1).to(device) # [B, 2, 3]
        # Just as neural body concept
        if render_kwargs_train['enlarge_box'] > 0.:
            bbox[:, 0, :] = bbox[:, 0, :] - render_kwargs_train['enlarge_box'] # swheo: for me, it was 0.2
            bbox[:, 1, :] = bbox[:, 1, :] + render_kwargs_train['enlarge_box']
        else:
            bbox[:, 0, 2] = bbox[:, 0, 2] - 0.05
            bbox[:, 1, 2] = bbox[:, 1, 2] + 0.05

        bounding_box = rend_util.get_2dmask_from_bbox(bbox, intrinsics, c2w, H, W)[0]
        # mask = model_input['object_mask'].to(device)
        # bounding_box = model_input['bbox_mask'].to(device)
        # near, far, valididx = rend_util.near_far_from_bbox(rays_o, rays_d, bounding_box)
        self.model.mesh.update_vertices(vertices[0]) #todo enable batched update
        if args.model.input_type in ['tframe','directproj','dispproj','disptframe']:
            self.model.tposeInfo = {key: val[0].to(device) for key, val in model_input['tposeInfo'].items()}
        elif args.model.input_type == 'invskin':
            self.model.transInfo = {key: val[0].to(device) for key, val in model_input['transformInfo'].items()}
        # N_rays=-1 for rendering full image
        if render_kwargs_train['strict_bbox_sampling']:
            # N_rays=-1 for rendering full image,
            rays_o, rays_d, select_inds, near, far = \
                rend_util.get_rays_nb(c2w, intrinsics, H, W, bbox, args.data.N_rays,
                                      jittered=args.training.jittered if args.training.get('jittered') is not None else False,
                                      mask=bounding_box.reshape([-1, H, W]) if args.training.get('sample_maskonly', False) is not False else None
                                      )
        else:
            rays_o, rays_d, select_inds = rend_util.get_rays(
                c2w, intrinsics, H, W, N_rays=args.data.N_rays,
                jittered=args.training.jittered if args.training.get('jittered') is not None else False,
                mask=bounding_box.reshape([-1, H, W])[0] if args.training.get('sample_maskonly',
                                                                              False) is not False else None  # ,
            )
            near, far = None, None


        # rays_o, rays_d, select_inds = rend_util.get_rays(
        #     c2w, intrinsics, H, W, N_rays=args.data.N_rays,
        #     jittered=args.training.jittered if args.training.get('jittered') is not None else False,
        #     mask=bounding_box.reshape([-1, H, W])[0] if args.training.get('sample_maskonly',
        #                                                                   False) is not False else None  # ,
        # )
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
        cos_anneal_ratio = get_cos_anneal_ratio(it, args.training.anneal_end) if args.training.anneal_end > 0. else -1.0
        rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=detailed_output,
                                             cbfeat_map=cbuvmap,
                                             idfeat_map=idfeatmap,
                                             smpl_param=smplparams,
                                             bounding_box=bbox.to(device)
                                             if args.training.get('sample_maskonly', False) is not False else None,
                                             cos_anneal_ratio=cos_anneal_ratio,
                                             **render_kwargs_train)

        # [B, N_rays, N_pts, 3]
        nablas: torch.Tensor = extras['implicit_nablas']
        # [B, N_rays, N_pts]
        nablas_norm = torch.norm(nablas, dim=-1)
        # [B, N_rays]
        mask_volume: torch.Tensor = extras['mask_volume']
        # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
        # Note. swheo, : dueto binary cross entropy (log(0) = -inf) - exploding
        # mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
        mask_volume = torch.clamp(mask_volume, 1e-3, 1 - 1e-3)
        extras['mask_volume_clipped'] = mask_volume

        losses = OrderedDict()

        # [B, N_rays, 3]
        losses['loss_img'] = F.l1_loss(rgb, target_rgb, reduction='none')
        # [B, N_rays, N_pts]
        losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm,
                                                                      nablas_norm.new_ones(nablas_norm.shape),
                                                                      reduction='mean')
        if 'logit' in extras:
            # [B, N_rays,]
            losses['loss_seg'] = args.training.w_seg * F.cross_entropy(extras['logit'].view((-1, 4)),
                                                                       target_segm.view((-1,)), reduction='mean')

        if args.training.with_mask:
            # [B, N_rays]
            target_mask = torch.gather(mask.to(device), 1, select_inds)
            losses['loss_mask'] = args.training.w_mask * F.binary_cross_entropy(mask_volume, target_mask.float(),
                                                                                reduction='mean')
            if mask_ignore is not None:
                target_mask = torch.logical_and(target_mask, mask_ignore)
            # [N_masked, 3]
            losses['loss_img'] = (losses['loss_img'] * target_mask[..., None].float()).sum() / (
                        target_mask.sum() + 1e-10)
            # losses['loss_seg'] = (losses['loss_seg'] * target_mask[0].float()).sum() / (target_mask.sum()+1e-10)
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
        extras['implicit_nablas_norm'] = nablas_norm
        extras['scalars'] = {'1/s': 1. / self.model.decoder.forward_s().data}
        extras['select_inds'] = select_inds
        # if detailed_output:
        #     sample_range = extras.pop('sample_range').reshape((-1, 2, 3))
        #     range_min = sample_range[:, 0, :].min(0)[0]
        #     range_max = sample_range[:, 1, :].max(0)[0]
        #     extras['scalars'].update(
        #         {'minx': range_min[0], 'miny': range_min[1], 'minz': range_min[2]})
        #     extras['scalars'].update(
        #         {'maxx': range_max[0], 'maxy': range_max[1], 'maxz': range_max[2]})

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
        cbuvmap = val_in['cbuvmap'].to(device)
        subject_id = val_in['subject_id'].item() if type(val_in['subject_id']) == torch.Tensor else val_in['subject_id']
        idfeatmap = render_kwargs_test['idfeat_map_all'][subject_id][None].to(device) \
            if render_kwargs_test['idfeat_map_all'] != None else None
        smplparams = val_in['smpl_params']['poses'][:, 0, 3:].to(device) if args.data.smpl_feat != 'none' else None
        vertices = val_in['vertices'].to(device)
        self.model.mesh.update_vertices(vertices[0]) #todo enable batched update
        # self.model.tposeInfo = {key: val[0].to(device) for key, val in val_in['tposeInfo'].items()}
        if args.model.input_type in ['tframe','directproj','dispproj','disptframe']:
            self.model.tposeInfo = {key: val[0].to(device) for key, val in val_in['tposeInfo'].items()}
        elif args.model.input_type == 'invskin':
            self.model.transInfo = {key: val[0].to(device) for key, val in val_in['transformInfo'].items()}

        mask_ = dilation(val_in['object_mask'].unsqueeze(0).to(device), torch.ones(11,11).to(device),
                         border_type='constant', border_value=0.)
        mask = mask_.reshape(B, -1) > 0.
        # Computed object bounding bbox
        # margin = 0.2
        # bbox = torch.stack([vertices.min(dim=0).values-margin, vertices.max(dim=0).values+margin]).to(device)
        # bounding_box = rend_util.get_2dmask_from_bbox(bbox, intrinsics[0], c2w[0], H, W)[0]

        bbox = torch.stack([vertices.min(dim=1).values, vertices.max(dim=1).values], dim=1).to(device) # [B, 2, 3]
        # Just as neural body concept
        if render_kwargs_test['enlarge_box'] > 0.:
            bbox[:, 0, :] = bbox[:, 0, :] - render_kwargs_test['enlarge_box'] # swheo: for me, it was 0.2
            bbox[:, 1, :] = bbox[:, 1, :] + render_kwargs_test['enlarge_box']
        else:
            bbox[:, 0, 2] = bbox[:, 0, 2] - 0.05
            bbox[:, 1, 2] = bbox[:, 1, 2] + 0.05

        if not render_kwargs_test['maskonly']:
            mask = None
            # margin = 0.0
            # bbox = torch.stack([vertices.min(dim=0).values-margin, vertices.max(dim=0).values+margin]).to(device)
            # mask = (rend_util.get_2dmask_from_bbox(bbox, intrinsics[0], c2w[0], H, W) > 0.).view(B, -1)

        # N_rays=-1 for rendering full image
        if render_kwargs_test['strict_bbox_sampling']:
            # N_rays=-1 for rendering full image,
            rays_o, rays_d, select_inds, near, far = \
                rend_util.get_rays_nb(c2w, intrinsics, H, W, bbox, -1, mask=mask)
        else:
            rays_o, rays_d, select_inds = rend_util.get_rays(
                c2w, intrinsics, H, W, N_rays=-1, mask=mask)
            near, far = None, None


        target_rgb = val_gt['rgb'].to(device)
        target_segm = val_gt['segm'].to(device)
        cos_anneal_ratio = get_cos_anneal_ratio(it, args.training.anneal_end) if args.training.anneal_end > 0. else -1.0
        rgb, depth_v, ret = self.renderer(rays_o, rays_d, detailed_output=False,
                                          calc_normal=True,
                                          cbfeat_map=cbuvmap,
                                          idfeat_map=idfeatmap,
                                          smpl_param=smplparams,
                                          bounding_box=bbox.to(device) if args.training.get('sample_maskonly',False) is not False else None,
                                          cos_anneal_ratio=cos_anneal_ratio,
                                          near=near,
                                          far=far,
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
        if 'normals_volume' in ret:
            val_imgs['val/predicted_normals'] = to_img(ret['normals_volume'] / 2. + 0.5, mask=mask)

        return val_imgs

    def val_mesh(self,
                 args,
                 val_in,
                 filepath,
                 render_kwargs_test: dict,
                 device='cuda'
                 ):
        cbuvmap = val_in['cbuvmap'][None].to(device)
        subject_id = val_in['subject_id'].item() if type(val_in['subject_id']) == torch.Tensor else val_in['subject_id']
        idfeatmap = render_kwargs_test['idfeat_map_all'][subject_id][None].to(device) \
            if render_kwargs_test['idfeat_map_all'] != None else None
        smplparams = val_in['smpl_params']['poses'][0, 3:].to(device) if args.data.smpl_feat != 'none' else None
        vertices = val_in['vertices'].to(device)
        # vertices = self.model.tposeInfo['vertices'] #todo delete this later
        self.model.mesh.update_vertices(vertices)
        if args.model.input_type in ['tframe','directproj','dispproj','disptframe']:
            self.model.tposeInfo = {key: val.to(device) for key, val in val_in['tposeInfo'].items()}
        elif args.model.input_type == 'invskin':
            self.model.transInfo = {key: val.to(device) for key, val in val_in['transformInfo'].items()}
        cbfeat, idGfeat, idCfeat, idSfeat = self.model.forward_featext(cbuvmap=cbuvmap, iduvmap=idfeatmap)

        # todo more sophisticated method?
        class fooFunc:
            def __init__(self, model):
                self.surface_func = functools.partial(model.forward_sdf,
                                                      cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat,
                                                      smpl_param=smplparams)

            def forward(self, x):
                return self.surface_func(x)[0]

        # todo enable bbox based acceleration

        implicit_surface = fooFunc(self.model)
        mesh_util.extract_mesh(
            implicit_surface,
            filepath=filepath,
            volume_size=args.data.get('volume_size', 2.0),
            chunk=16*1024,
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
        # --------------
        # pretrain sdf using IGR's framework
        # https://www.github.com/amosgropp/IGR
        # --------------

        from torch import optim
        sigma_global = args.model.decoder.obj_bounding_radius / 2.
        params = list(self.model.decoder.implicit_surface.parameters())
        params += list(self.model.cbfeat_layer.parameters()) if self.model.cbfeat_layer is not None else []
        params += list(self.model.idfeat_layer.parameters()) if self.model.idfeat_layer is not None else []
        if render_kwargs_test['idfeat_map_all'] != None:
            optimizer = optim.Adam([{"params":params, "lr":lr}, {"params":render_kwargs_test['idfeat_map_all'], "lr":args.training.lr_idfeat}])
        else:
            optimizer = optim.Adam(params, lr=lr)
        # todo : Use epoch - iteration not iteration only
        # swheo : As in https://github.com/jby1993/SelfReconCode/blob/538dcb24b90eb4f5412e6379ced027cc8153cdd0/model/network.py#L219
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 0.5)
        local_sigma = torch.from_numpy(dataloader.dataset.local_sigma).to(device).unsqueeze(-1).float()

        if it >= num_iters:
            return False, it

        with tqdm(range(num_iters)) as pbar:
            pbar.update(it)
            while it < num_iters:
                for (indices, model_input, ground_truth) in dataloader:
                    cbuvmap = model_input['cbuvmap'].to(device)
                    idfeatmap = render_kwargs_test['idfeat_map_all'][model_input['subject_id'].item()][None].to(device) \
                        if render_kwargs_test['idfeat_map_all'] != None else None
                    smpl_param = model_input['smpl_params']['poses'][:, 0, :].to(
                        device) if args.data.smpl_feat != 'none' else None
                    vertices = model_input['vertices'][0].to(device)
                    if args.model.input_type in ['tframe','directproj','dispproj','disptframe']:
                        self.model.tposeInfo = {key: val[0].to(device) for key, val in model_input['tposeInfo'].items()}
                        # vertices = self.model.tposeInfo['vertices'] # to check pose deviations
                    elif args.model.input_type == 'invskin':
                        self.model.transInfo = {key: val[0].to(device) for key, val in model_input['transformInfo'].items()}

                    # vertices = self.model.tposeInfo['vertices']
                    sample_inds = torch.randint(vertices.shape[0], (batch_points,), device=device)
                    # Sampling on body vertices
                    pts_on = vertices[sample_inds, :]
                    # Sampling other points
                    pts_out_loc = pts_on + torch.randn_like(pts_on) * local_sigma[sample_inds, :]
                    pts_out_glo = torch.rand(vertices.shape[0] // 2, 3, device=device) * (
                            sigma_global * 2) - sigma_global
                    pts_out = torch.cat([pts_out_loc, pts_out_glo], dim=0)
                    # pts_out = torch.load('/ssd3/swheo/dev/code/neurecon/utils/pts_out.pt', map_location=pts_on.device) # todo tmp
                    self.model.mesh.update_vertices(vertices)
                    normals = self.model.mesh.vert_normal[sample_inds, :]
                    cbfeat, idGfeat, idCfeat, idSfeat = self.model.forward_featext(cbuvmap=cbuvmap, iduvmap=idfeatmap)

                    pts_ = torch.cat([pts_on, pts_out], dim=0)
                    sdf_, nablas_, _, _ = self.model.forward_sdf_with_nablas(pts_.unsqueeze(0), cbfeat=cbfeat,
                                                                             idGfeat=idGfeat,
                                                                             idCfeat=idCfeat,
                                                                             idSfeat=idSfeat,
                                                                             smpl_param=smpl_param)
                    sdf_on = sdf_[..., :pts_on.shape[0]]
                    nablas_on = nablas_[..., :pts_on.shape[0], :]
                    nablas_out = nablas_[..., pts_on.shape[0]:, :]
                    # todo could add sdf loss since we compute distance
                    # manifold loss
                    loss_ptson = (sdf_on.abs()).mean()

                    # eikonal loss
                    loss_eik = .1 * ((nablas_out.norm(2, dim=-1) - 1) ** 2).mean()

                    # normal loss
                    loss_norm = 1 * ((nablas_on - normals).abs()).norm(2, dim=-1).mean()

                    loss = loss_ptson + loss_eik + loss_norm

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norms = train_util.calc_grad_norm(model=self.model)
                    if grad_norms['total'] > 50:
                        foo = 1
                    optimizer.step()
                    it += 1
                    if logger is not None:
                        logger.add('pretrain_sdf', 'loss', loss.item(), it)
                        logger.add('pretrain_sdf', 'loss_pts', loss_ptson.item(), it)
                        logger.add('pretrain_sdf', 'loss_eik', loss_eik.item(), it)
                        logger.add('pretrain_sdf', 'loss_norm', loss_norm.item(), it)

                    pbar.set_postfix(loss_total=loss.item(), loss_pts=loss_ptson.item(), loss_eik=loss_eik.item(),
                                     loss_norm=loss_norm.item(), grad_norm=grad_norms['total'])
                    pbar.update(1)
                    if it >= num_iters:
                        break

        return True, it