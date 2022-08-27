from models.frameworks.neussegm import NeuSSegm
from models.base import ImplicitSurface, NeRF, RadianceNet, forward_with_nablas
from models.layers.pop_backbone import get_unet_backbone
from models.trainers.trainer_garmnerf import Trainer
from utils import rend_util, train_util, mesh_util, io_util


from zju_smpl.body_model import SMPLlayer
from utils.geometry import Mesh, project2closest_face, texcoord2imcoord
from utils.dispproj import DispersedProjector, _approx_inout_sign_by_normals

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

def sample_feature_volume_uv(uv, feat_volume, splits=[]):
    """
    :param uv : uv point(point in uvmap) [B, N_ryas, N_pts, 2]
    :param feat_volume : feature volume, [B, ch, H, W]
    :param splits : list indicate split of feature volume if needed
    """
    uv_ = uv * 2. - 1.

    # Make dimensions to be square
    # Make uv resides in [-1,1] to be compatible with grid_sample
    # Note. padding dimension starts from end, ((lastdim start, lastdim end), ..., (firstdim start, firstdim end)) order
    # paddim = (0, 0, 0, uv_.shape[2] % 2, 0, uv_.shape[1] % 2, 0, 0)
    # uv_pad = F.pad(uv_, paddim, mode='constant', value=-1.)

    # Convert it to [B, N_ray, N_sample, Channel] by permute
    # sample_feat = F.grid_sample(feat_volume, uv_, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
    sample_feat = train_util.grid_sample(feat_volume, uv_).permute(0, 2, 3, 1)

    return sample_feat if len(splits) == 0 else sample_feat[:, :uv.shape[1], :uv.shape[2], :].split(splits, dim=-1)


asset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../assets/"
smpl_dir = '../../assets/smpl/smpl'

class GarmentNerf(nn.Module):
    def __init__(self,
                 W_idG_feat=32,
                 W_idC_feat=32,
                 W_idS_feat=32,
                 cbfeat_cfg=None,  # Canonical Body Feature
                 idfeat_cfg=None,
                 input_type='xyz',
                 decoder_cfg=dict(),
                 **kwargs):
        super().__init__()
        """
        cbfeat : unet / input_ch, output_nc, nf, up_mode, use_dropout, return_lowers, return2branches
        """
        self.input_type = input_type
        # Canonical body feature module
        # todo, make it compatible with other type (e.g., Graph Conv or Xformer arch.)
        if idfeat_cfg != None:
            idfeat_name = idfeat_cfg.pop('layername')
            self.idfeat_layer = get_unet_backbone(idfeat_name, idfeat_cfg)
            self.W_idG_feat = W_idG_feat
            self.W_idC_feat = W_idC_feat
            self.W_idS_feat = W_idS_feat
        else:
            self.idfeat_layer = None

        # todo, add pose embedding layer
        if cbfeat_cfg != None:
            cbfeat_name = cbfeat_cfg.pop('layername')
            self.cbfeat_layer = get_unet_backbone(cbfeat_name, cbfeat_cfg)
        else:
            self.cbfeat_layer = None

        # self.resenc = ImplicitSurface(W=256, D=4, input_dim=3, output_dim=3, radius_init=0.0, obj_bounding_size=0.0, embed_multires=6,
        #          W_geo_feat=-1, skips=[], featcats=[], W_up=[], geometric_init=False, weight_norm=True, use_siren=False)

        self.decoder = NeuSSegm(**decoder_cfg)


        # geom_proc_layers = {
        #     'unet': UnetNoCond5DS(c_geom, c_geom, nf, up_mode, use_dropout), # use a unet
        #     'conv': GeomConvLayers(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
        #     'bottleneck': GeomConvBottleneckLayers(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
        #     'gaussian': GaussianSmoothingLayers(channels=c_geom, kernel_size=gaussian_kernel_size, sigma=1.0), # use a fixed gaussian smoother
        # }
        # unets = {32: UnetNoCond5DS, 64: UnetNoCond6DS, 128: UnetNoCond7DS, 256: UnetNoCond7DS}
        #
        self.mesh = Mesh(file_name=os.path.join(asset_dir, 'smpl/smpl/smpl_uv.obj'))
        self.tposeInfo = None
        self.transInfo = None
        if self.input_type == 'dispproj':
            self.dispprojfunc = DispersedProjector(cache_path='assets/smpl/smpl', mesh=self.mesh)
            # todo check Author's original impl
            # from utils.Disperse_projection_ref import SurfaceAlignedConverter
            # self.dispprojfunc_ = SurfaceAlignedConverter(verts=self.mesh.vertices, faces=self.mesh.faces, device=self.mesh.device,
            #                                             cache_path='assets/smpl/smpl')

        # uvmap = torch.from_numpy(load_rgb(os.path.join(asset_dir, 'smpl/smpl/smpl_uv.png'), downscale=4))
        # self.uvmask = uvmap[-1]
        # self.uvmap = uvmap[:-1] * self.uvmask.unsqueeze(0)

    def to(self, device):
        new_self = super(GarmentNerf, self).to(device)
        new_self.mesh.to(device)
        if self.input_type=='dispproj':
            new_self.dispprojfunc.to(device)
            # new_self.dispprojfunc_.to(device)

        return new_self

    def forward_featext(self, cbuvmap: torch.Tensor, iduvmap: torch.Tensor):
        """
        :param cbuvmap : canonical body uvmap feature, [B, ch, H, W]
        :param iduvmap : identity uvmap feature, [B, ch, H, W]
        """

        # canonical body feature
        cbfeat = self.cbfeat_layer(cbuvmap) if self.cbfeat_layer is not None else None

        # Diffuse identity feature if required
        if iduvmap is None:
            idGfeat, idCfeat, idSfeat = None, None, None
        else:
            idGfeat, idCfeat, idSfeat = iduvmap.split([self.W_idG_feat, self.W_idC_feat, self.W_idS_feat], dim=1)
            if self.idfeat_layer is not None:
                idGfeat = self.idfeat_layer(idGfeat) if idGfeat.shape[1] != 0 else None
                idCfeat = self.idfeat_layer(idCfeat) if idCfeat.shape[1] != 0 else None
                idSfeat = self.idfeat_layer(idSfeat) if idSfeat.shape[1] != 0 else None

        return cbfeat, idGfeat, idCfeat, idSfeat

    def forward_sdf(self, x: torch.Tensor,
                    cbfeat: torch.Tensor, idGfeat: torch.Tensor, idCfeat: torch.Tensor, idSfeat: torch.Tensor,
                    smpl_param: torch.Tensor,
                    return_h=False):
        """
        :param x : xyz points, [B, N_ryas, N_pts, 3]
        :param cbfeat : canonical body uvmap feature, [B, ch, H, W]
        :param idGfeat : identity uvmap feature, [B, self.W_idG_feat, H, W]
        :param idCfeat : identity uvmap feature, [B, self.W_idC_feat, H, W]
        :param idSfeat : identity uvmap feature, [B, self.W_idC_feat, H, W]
        :param smpl_param : smpl_parameter(Not decided to use shape or poses) [B, N_params]
        """
        ## Compute uvh (uv loc / signed distance from surface) No..
        # Note. sign means whether it the line between x to surface is aligned (+) or reverse (-)
        if len(x.shape)<3:
            x = x.unsqueeze(0)

        vnear, st, idxs = project2closest_face(x, self.mesh, stability_scale = 50.) #todo use argument!

        if self.input_type == 'xyz':
            uvh = x
        elif self.input_type == 'uvh':
            uv = self.mesh.get_uv_from_st(st, idxs)
            uv_ = texcoord2imcoord(uv, 2, 2)
            trinorms = self.mesh.faces_normal[idxs]
            x_diff = x - vnear
            # Note. To prevent Nan, use square distance rather than using sqrt()
            x_dists_square = torch.sqrt(torch.sum(x_diff**2, -1)+1e-8).unsqueeze(-1)
            x_sign = torch.sign(torch.sum(trinorms * x_diff, dim=-1)).unsqueeze(-1)
            uvh = torch.cat([uv_, x_dists_square * x_sign], dim=-1)
        elif self.input_type == 'tframe':
            # Canonical space alignment method
            x_diff = x - vnear
            x_tf = self.mesh.faces_tanframe[idxs] @ x_diff.unsqueeze(-1)
            vnear_tpose = self.tposeInfo['B'][idxs] + \
                          st[...,0].unsqueeze(-1)*self.tposeInfo['E0'][idxs] + \
                          st[...,1].unsqueeze(-1)*self.tposeInfo['E1'][idxs]
            # x_tpose = (self.tposeInfo['tanframe'][idxs].transpose(-1, -2) @ x_tf)[...,0] + vnear_tpose
            x_tpose = (self.tposeInfo['tanframe_inv'][idxs] @ x_tf)[..., 0] + vnear_tpose
            # x_sign = torch.tanh(100.0 * torch.sum(trinorms * x_diff, dim=-1).unsqueeze(-1))
            # uvh = x_tpose  # + self.resenc(x).unsqueeze(1)
            uvh = torch.cat([x_tpose, vnear_tpose], dim=-1)
        elif self.input_type == 'invskin':
            # Move to smpl object(canonical)  space
            W = self.transInfo['W']
            x_hom = torch.concat([x, torch.ones(x[...,0].shape, device=x.device).unsqueeze(-1)], dim=-1)
            x_can = (x_hom@self.transInfo['alignMat'])[...,:3]
            WB = W[self.mesh.faces][idxs, 0]
            WE0 = (W[self.mesh.faces][:, 1, :] - W[self.mesh.faces][:, 0, :])[idxs]
            WE1 = (-W[self.mesh.faces][:, 0, :] + W[self.mesh.faces][:, 2, :])[idxs]
            W_q = (WB + st[..., 0].unsqueeze(-1) * WE0 + st[..., 1].unsqueeze(-1) * WE1).view(-1, 24)
            W_q = W_q/torch.sum(W_q+1e-8, dim=-1).unsqueeze(-1)
            T = (W_q[None].expand([1,-1,-1])@self.transInfo['A'].contiguous().view(1, W.shape[-1], 16)).view(1, -1, 4, 4)
            q_hom = torch.cat([x_can.view(-1,3), torch.ones_like(x_can.view(-1,3)[...,0])[...,None]], dim=-1)
            q_inv = torch.linalg.inv(T)@(q_hom.view(-1,4,1))
            q_inv = q_inv[...,:3, 0].view(x.shape)
            uvh = q_inv
        elif self.input_type == 'directproj':
            # todo dispersed projection
            vnear_tpose = self.tposeInfo['B'][idxs] + \
                          st[..., 0].unsqueeze(-1) * self.tposeInfo['E0'][idxs] + \
                          st[..., 1].unsqueeze(-1) * self.tposeInfo['E1'][idxs]

            x_sign = _approx_inout_sign_by_normals(x, vnear,
                                                  torch.cat([(1 - torch.sum(st, -1)).unsqueeze(-1), st], dim=-1)
                                                  , self.mesh.vert_normal[self.mesh.faces][idxs]).unsqueeze(-1)
            x_diff = x - vnear
            # x_dists = torch.norm(x_diff, dim=-1).unsqueeze(-1)
            x_dists = torch.sqrt(torch.sum(x_diff ** 2, -1) + 1e-12).unsqueeze(-1)

            uvh = torch.cat([vnear_tpose, x_dists * x_sign], dim=-1)
            # debug
            # from dataio.MviewTemporalSMPL import plotly_viscorres3D
            # plotly_viscorres3D(vnear[..., :1, :].detach().cpu(), x[..., :1, :].detach().cpu(),
            #                    x_diff[..., :1, :].detach().cpu(), self.mesh.vertices.detach().cpu(),
            #                    self.mesh.faces.detach().cpu(), faces_tanframe=None)
        elif self.input_type == "dispproj":
            # # todo keep Author's original impl
            # if len(x.shape) == 3:
            #     N_ray = 1
            #     B, N_pts, _ = x.shape
            # elif len(x.shape) == 4:
            #     B, N_ray, N_pts, _ = x.shape
            # else:
            #     N_ray, B = 1, 1
            #     N_pts, _ = x.shape
            #
            # uvh_, nearest, nearest_new, barycentric = self.dispprojfunc_.xyz_to_xyzch(x.view(1,-1,3),
            #                                                              self.mesh.vertices.unsqueeze(0),
            #                                                              xyzc_in=self.tposeInfo['vertices'], debug=True)
            # uvh_ = uvh_.view((B, N_ray, N_pts, -1))

            # Dispersed projection
            h, st_, _, vnear_, idxs_ = self.dispprojfunc(x, self.mesh, vnear=vnear, st=st, idxs=idxs)
            vnear_tpose = self.tposeInfo['B'][idxs_] + \
                          st_[..., 0].unsqueeze(-1) * self.tposeInfo['E0'][idxs_] + \
                          st_[..., 1].unsqueeze(-1) * self.tposeInfo['E1'][idxs_]
            st = st_
            idxs = idxs_
            uvh = torch.cat([vnear_tpose, h], dim=-1)

            # debug
            # from dataio.MviewTemporalSMPL import plotly_viscorres3D
            # numpts = 5
            # plotly_viscorres3D(vertices=self.mesh.vertices.detach().cpu(), faces=self.mesh.faces.detach().cpu(),
            #                    query=x[..., :numpts, :].detach().cpu(),
            #                    vnear=vnear_[..., :numpts, :].detach().cpu(), pp_color=vnear[..., :numpts, :].detach().cpu(),
            #                    faces_tanframe=None)
            # plotly_viscorres3D(vertices=self.tposeInfo['vertices'].detach().cpu(), faces=self.mesh.faces.detach().cpu(),
            #                    query=x[..., :numpts, :].detach().cpu(),
            #                    vnear=vnear_tpose[..., :numpts, :].detach().cpu(), pp_color=vnear[..., :numpts, :].detach().cpu(),
            #                    faces_tanframe=None)
        else:
            raise NotImplementedError

        uv = self.mesh.get_uv_from_st(st, idxs)
        uv_ = texcoord2imcoord(uv, 2, 2)
        # # Align to each coordinate frame
        aggrfeat = torch.Tensor(0).to(uvh.device)  # dummy tensor
        splits = []
        for feat in (cbfeat, idGfeat, idCfeat, idSfeat):
            if feat is not None:
                aggrfeat = torch.cat([aggrfeat, feat], 1)  # [B, C, H, W]
                splits.append(feat.shape[1])
        sample_feat = list(sample_feature_volume_uv(uv_, aggrfeat,
                                                    splits=splits)) if len(splits) > 0 else []
        if len(x.shape) < 4:
            # For batchfying
            uvh = uvh[:, 0, ...] if len(uvh.shape)==4 else uvh #todo for uvh and others
            sample_feat = [sample_feat_[:, 0, ...] for sample_feat_ in sample_feat]
            smpl_param = smpl_param.expand(x.shape[0], x.shape[1], -1) if smpl_param is not None \
                else torch.Tensor(0).to(uvh.device)
        else:
            smpl_param = smpl_param.expand(x.shape[0], x.shape[1], x.shape[2], -1) if smpl_param is not None \
                else torch.Tensor(0).to(uvh.device)
        cbfeat_ = torch.Tensor(0).to(uvh.device) if cbfeat is None else sample_feat.pop(0)
        idGfeat_ = torch.Tensor(0).to(uvh.device) if idGfeat is None else sample_feat.pop(0)
        idCfeat_ = torch.Tensor(0).to(uvh.device) if idCfeat is None else sample_feat.pop(0)
        idSfeat_ = torch.Tensor(0).to(uvh.device) if idSfeat is None else sample_feat.pop(0)

        input_s = torch.cat([cbfeat_, idGfeat_, smpl_param, uvh], dim=-1)  # [B, N_ray, N_sample, C]
        feats = (cbfeat_, idGfeat_, idCfeat_, idSfeat_, uvh)  # [B, N_ray, N_sample, C]
        if return_h:
            sdf, h = self.decoder.implicit_surface.forward(input_s, return_h=return_h)
            return sdf, feats, h
        else:
            sdf = self.decoder.implicit_surface.forward(input_s, return_h=return_h)
            return sdf, feats

    def forward_sdf_with_nablas(self, x: torch.Tensor,
                                cbfeat: torch.Tensor, idGfeat: torch.Tensor, idCfeat: torch.Tensor, idSfeat: torch.Tensor,
                                smpl_param: torch.Tensor, ):
        """
        :param x : xyz points, [B, N_ryas, N_pts, 3]
        :param cbfeat : canonical body uvmap feature, [B, ch, H, W]
        :param idGfeat : identity uvmap feature, [B, self.W_idG_feat, H, W]
        """
        func = functools.partial(self.forward_sdf, cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat,
                                 smpl_param=smpl_param,
                                 return_h=True)
        sdf, nablas, feats, h = forward_with_nablas(func, x)
        return sdf, nablas, feats, h

    def forward_radiance(self, x: torch.Tensor, view_dirs: torch.Tensor,
                         cbfeat: torch.Tensor, idGfeat: torch.Tensor, idCfeat: torch.Tensor, idSfeat: torch.Tensor,
                         smpl_param: torch.Tensor
                         ):
        """
        :param x : xyz points, [B, N_ryas, N_pts, 3]
        :param view_dirs : uv point and distance [B, N_ryas, N_pts, 3]
        :param cbfeat : canonical body uvmap feature, [B, ch, H, W]
        :param idGfeat : identity uvmap feature, [B, self.W_idG_feat, H, W]
        :param idCfeat : identity uvmap feature, [B, self.W_idC_feat, H, W]
        """
        _, nablas, feats, intmed_feat = self.forward_sdf_with_nablas(x,
                                                                     cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat,
                                                                     smpl_param=smpl_param)
        # todo rename uvh
        # feats[-3] : idCfeat_, feates[-1]: uvh
        x_ = torch.cat([feats[-3], feats[-1]], dim=-1)  # concat with color identity feature, uvh
        radiances = self.decoder.radiance_net.forward(x_, view_dirs, nablas, intmed_feat)
        return radiances

    def forward_segm(self, x: torch.Tensor, view_dirs: torch.Tensor,
                     cbfeat: torch.Tensor, idGfeat: torch.Tensor, idCfeat: torch.Tensor, idSfeat: torch.Tensor,
                     smpl_param: torch.Tensor
                     ):
        _, feats, intmed_feat = self.forward_sdf(x, cbfeat=cbfeat, idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat,
                                                   smpl_param=smpl_param,
                                                   return_h=True)
        # feats[-2] : idCfeat_, feates[-1]: uvh
        x_ = torch.cat([feats[-2], feats[-1]], dim=-1)  # concat with color identity feature, uvh
        logits = self.decoder.segm_net.forward(x_, view_dirs, view_dirs,
                                               intmed_feat)  # Note. second, third input will not be used
        return logits

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor,
                cbuvmap: torch.Tensor, iduvmap: torch.Tensor, smpl_param: torch.Tensor):
        """
        :param x : xyz points, [B, N_ryas, N_pts, 3]
        :param uvh : uv point and distance [B, N_ryas, N_pts, 3]
        :param view_dirs : uv point and distance [B, N_ryas, N_pts, 3]
        :param cbuvmap : canonical body uvmap feature, [B, ch, H, W]
        :param iduvmap : identity uvmap feature, [B, ch, H, W]
        """
        cbfeat, idGfeat, idCfeat, idSfeat = self.forward_featext(cbuvmap, iduvmap)
        sdf, nablas, feats, intmed_feat = self.forward_sdf_with_nablas(x, cbfeat=cbfeat, idGfeat=idGfeat,
                                                                       idCfeat=idCfeat, idSfeat=idSfeat,
                                                                       smpl_param=smpl_param)
        x_ = torch.cat([feats[-3], feats[-1]], dim=-1)  # concat with color identity feature, uvh
        radiances = self.decoder.radiance_net.forward(x_, view_dirs, nablas, intmed_feat)
        x_ = torch.cat([feats[-2], feats[-1]], dim=-1)  # concat with color identity feature, uvh
        logits = self.decoder.segm_net.forward(x_, view_dirs, view_dirs, intmed_feat)
        return radiances, sdf, nablas, logits


def get_model(args):
    model_config = {
        # todo make the W_idG_feat and W_idC_feat same
        'input_type': args.model.setdefault('input_type', 'xyz'),
        'W_idG_feat': args.model.setdefault('W_idG_feat', 32),
        'W_idC_feat': args.model.setdefault('W_idC_feat', 32),
        'W_idS_feat': args.model.setdefault('W_idS_feat', 32),
        'use_segm'  : args.model.setdefault('use_segm',True),
        'use_cbfeat': args.model.setdefault('use_cbfeat', True),
        'use_idfeat': args.model.setdefault('use_idfeat', True)
    }


    decoder_cfg = {
        'obj_bounding_radius': args.model.decoder.obj_bounding_radius,
        # 'input_ch': args.model.decoder.setdefault('input_ch', 3),
        'W_geo_feat': args.model.decoder.setdefault('W_geometry_feature', 256),
        'use_outside_nerf': args.model.decoder.setdefault('use_outside_nerf', False),
        'speed_factor': args.training.setdefault('speed_factor', 1.0),
        'variance_init': args.model.decoder.setdefault('variance_init', 0.05)
    }

    if model_config['input_type'] in ['directproj', 'dispproj']:# 'tframe'
        decoder_cfg['input_ch'] = 4
    elif model_config['input_type'] == 'tframe':
        decoder_cfg['input_ch'] = 6
    else:
        decoder_cfg['input_ch'] = 3

    surface_cfg = {
        'embed_multires': args.model.decoder.surface.setdefault('embed_multires', 6),
        'radius_init': args.model.decoder.surface.setdefault('radius_init', 1.0),
        'geometric_init': args.model.decoder.surface.setdefault('geometric_init', True),
        'D': args.model.decoder.surface.setdefault('D', 8),
        'W': args.model.decoder.surface.setdefault('W', 256),
        'W_up': args.model.decoder.surface.setdefault('W_up', []),
        'skips': args.model.decoder.surface.setdefault('skips', [4]),
        'featcats': args.model.decoder.surface.setdefault('featcats', []),
        'input_feat': 0
    }

    radiance_cfg = {
        'embed_multires': args.model.decoder.radiance.setdefault('embed_multires', -1),
        'embed_multires_view': args.model.decoder.radiance.setdefault('embed_multires_view', -1),
        'use_view_dirs': args.model.decoder.radiance.setdefault('use_view_dirs', True),
        'D': args.model.decoder.radiance.setdefault('D', 4),
        'W': args.model.decoder.radiance.setdefault('W', 256),
        'W_up': args.model.decoder.radiance.setdefault('W_up', []),
        'skips': args.model.decoder.radiance.setdefault('skips', []),
        'featcats': args.model.decoder.radiance.setdefault('featcats', []),
        'input_feat': 0
    }

    if model_config['use_segm']:
        segmentation_cfg = {
            'embed_multires': args.model.decoder.segmentation.setdefault('embed_multires', -1),
            'embed_multires_view': args.model.decoder.segmentation.setdefault('embed_multires_view', -1),
            'use_view_dirs': args.model.decoder.segmentation.setdefault('use_view_dirs', False),
            'D': args.model.decoder.segmentation.setdefault('D', 5),
            'W': args.model.decoder.segmentation.setdefault('W', 256),
            'W_up': args.model.decoder.segmentation.setdefault('W_up', []),
            'featcats': args.model.decoder.segmentation.setdefault('featcats', []),
            'skips': args.model.decoder.segmentation.setdefault('skips', []),
            'output_dim': args.model.decoder.segmentation.setdefault('output_dim', 4),
            'input_feat': 0
        }
    else:
        segmentation_cfg = None

    if model_config['use_cbfeat']:
        cbfeat_cfg = {
            'layername': args.model.canonicalbody.setdefault('layername', 'UnetNoCond5DS'),
            'input_nc': args.model.canonicalbody.setdefault('input_nc', 3),
            'nf': args.model.canonicalbody.setdefault('nf', 64),
            'output_nc': args.model.canonicalbody.setdefault('output_nc', 3),
            'up_mode': args.model.canonicalbody.setdefault('up_mode', 'upconv'),
            'use_dropout': args.model.canonicalbody.setdefault('use_dropout', False),
            'return_lowres': args.model.canonicalbody.setdefault('return_lowres', False),
            'return_2branches': args.model.canonicalbody.setdefault('return_2branches', False)
        }
        model_config['cbfeat_cfg'] = cbfeat_cfg
        input_feat_surface = cbfeat_cfg['output_nc']
    else:
        input_feat_surface = 0

    radiance_cfg['input_dim_pts'] = decoder_cfg['input_ch']
    if segmentation_cfg is not None:
        segmentation_cfg['input_dim_pts'] = decoder_cfg['input_ch']

    if model_config['use_idfeat']:
        idfeat_cfg = {
            'layername': args.model.identityfeat.setdefault('layername', 'GeomConvLayers'),
            'input_nc': model_config['W_idG_feat'],# + model_config['W_idC_feat'],
            'nf': args.model.identityfeat.setdefault('nf', 16),
            'output_nc': model_config['W_idG_feat'],# + model_config['W_idC_feat'],
        }
        #todo check W_idG_feat and others
        idfeat_len = model_config['W_idG_feat'] + model_config['W_idC_feat'] + model_config['W_idS_feat']
        idfeat_map = torch.ones(len(args.data.subjects),
                                idfeat_len, args.data.uv_size,
                                args.data.uv_size).normal_(
            mean=0., std=0.01).cuda()
        idfeat_map.requires_grad = True
        input_feat_surface += model_config['W_idG_feat']
        input_feat_radiance = model_config['W_idC_feat']
        input_feat_segmentation = model_config['W_idS_feat']
        radiance_cfg['input_feat'] = input_feat_radiance
        if segmentation_cfg is not None:
            segmentation_cfg['input_feat'] = input_feat_segmentation
        model_config['idfeat_cfg'] = idfeat_cfg
    else:
        idfeat_map = None

    if args.data.smpl_feat == 'pose':
        surface_cfg['input_feat'] = input_feat_surface + 72 - 3
    elif args.data.smpl_feat == 'beta':
        surface_cfg['input_feat'] = input_feat_surface + 10
    else:
        surface_cfg['input_feat'] = input_feat_surface

    decoder_cfg['surface_cfg'] = surface_cfg
    decoder_cfg['radiance_cfg'] = radiance_cfg
    decoder_cfg['segmentation_cfg'] = segmentation_cfg

    model_config['decoder_cfg'] = decoder_cfg

    model = GarmentNerf(**model_config)

    ## render kwargs

    render_kwargs_train = {
        # upsample config
        'upsample_algo': args.model.decoder.setdefault('upsample_algo', 'official_solution'),
        # [official_solution, direct_more, direct_use]
        'N_nograd_samples': args.model.decoder.setdefault('N_nograd_samples', 2048),
        'N_upsample_iters': args.model.decoder.setdefault('N_upsample_iters', 4),
        'N_outside': args.model.decoder.setdefault('N_outside', 0) if decoder_cfg['use_outside_nerf'] else 0,
        'with_mask': args.training.setdefault('with_mask', True),
        'obj_bounding_radius': args.model.decoder.setdefault('obj_bounding_radius', 1.0),
        'batched': args.data.batch_size is not None,
        'perturb': args.model.setdefault('perturb', True),  # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
        'idfeat_map_all': idfeat_map
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['idfeat_map_all'] = idfeat_map
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['maskonly'] = args.data.val_rendermaskonly
    render_kwargs_test['perturb'] = False

    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])

    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer

