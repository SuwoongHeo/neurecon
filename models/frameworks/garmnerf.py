from models.base import ImplicitSurface, NeRF, RadianceNet, forward_with_nablas
from utils import rend_util, train_util, mesh_util, io_util
from models.layers.pop_backbone import get_unet_backbone

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
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

import matplotlib.cm as colormap


def cdf_Phi_s(x, s):
    return torch.sigmoid(x * s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]

    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)

    # [(B), N_rays, N_pts-1]
    visibility_weights = opacity_alpha * \
                         torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ], dim=-1)

    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha * \
                         torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights


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

        self.decoder = NeusSegm(**decoder_cfg)


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
            x_dists = torch.norm(x_diff, dim=-1).unsqueeze(-1)

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

            uvh = torch.cat([vnear_tpose, h], dim=-1)

            # debug
            # from dataio.MviewTemporalSMPL import plotly_viscorres3D
            # numpts = 5
            # plotly_viscorres3D(self.mesh.vertices.detach().cpu(), self.mesh.faces.detach().cpu(),
            #                    query=x[..., :numpts, :].detach().cpu(),
            #                    vnear=vnear_[..., :numpts, :].detach().cpu(), pp_color=vnear[..., :numpts, :].detach().cpu(),
            #                    faces_tanframe=None)
            # plotly_viscorres3D(self.tposeInfo['vertices'].detach().cpu(), self.mesh.faces.detach().cpu(),
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


class NeusSegm(nn.Module):
    def __init__(self,
                 variance_init=0.05,
                 speed_factor=1.0,

                 input_ch=3,
                 W_geo_feat=-1,
                 use_outside_nerf=False,
                 obj_bounding_radius=1.0,

                 surface_cfg=dict(),
                 radiance_cfg=dict(),
                 segmentation_cfg=dict()):
        super().__init__()

        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor

        # ------- surface network
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_dim=input_ch, obj_bounding_size=obj_bounding_radius, **surface_cfg)

        # ------- radiance network
        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat, **radiance_cfg)
        if segmentation_cfg is not None:
            self.segm_net = RadianceNet(
                W_geo_feat=W_geo_feat, **segmentation_cfg)
            cmap = colormap.get_cmap('jet', segmentation_cfg['output_dim'])
            self.labels_cmap = torch.from_numpy(cmap(range(segmentation_cfg['output_dim']))[:, :3])
        else:
            self.segm_net = None

        # -------- outside nerf++
        if use_outside_nerf:
            enable_semantic = segmentation_cfg is not None
            semantic_dim = segmentation_cfg['output_dim'] if enable_semantic else 0
            self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True,
                                     enable_semantic=True, num_semantic_classes=semantic_dim)

    def forward_radiance(self, x: torch.Tensor, view_dirs: torch.Tensor):
        _, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiance = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiance

    def forward_segm(self, x: torch.Tensor, view_dirs: torch.Tensor):
        _, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        logits = self.segm_net.forward(x, view_dirs, nablas,
                                       geometry_feature)  # Note, View_dirs, nablas, are not used here
        return logits

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        logits = self.segm_net.forward(x, view_dirs, nablas,
                                       geometry_feature)  # Note, View_dirs, nablas, are not used here
        return radiances, sdf, nablas, logits


def volume_render(
        rays_o,
        rays_d,
        model: GarmentNerf,

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
        if bounding_box is None:
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
        # ------------------
        # Inside Scene
        # ------------------
        sdf, nablas, _, _ = batchify_query(model.forward_sdf_with_nablas, pts, cbfeat=cbfeat, idGfeat=idGfeat,
                                           idCfeat=idCfeat, idSfeat=idSfeat,
                                           smpl_param=smpl_param)
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
        if N_outside > 0:
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
            if bounding_box is None:
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


class SingleRenderer(nn.Module):
    def __init__(self, model: GarmentNerf):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


class Trainer(nn.Module):
    def __init__(self, model: GarmentNerf, device_ids=[0], batched=True):
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
        B, H, W = model_input['object_mask'].shape
        mask = model_input['object_mask'].reshape(B, -1).to(device)
        # H, W = model_input['H'].item(), model_input['W'].item()
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        cbuvmap = model_input['cbuvmap'].to(device)
        idfeatmap = render_kwargs_train['idfeat_map_all'][model_input['subject_id'].item()][None].to(device) \
            if render_kwargs_train['idfeat_map_all'] != None else None
        smplparams = model_input['smpl_params']['poses'][:, 0, 3:].to(device) if args.data.smpl_feat != 'none' else None
        vertices = model_input['vertices'][0].to(device)
        # Computed object bounding bbox
        margin = 0.1
        bbox = torch.stack([vertices.min(dim=0).values-margin, vertices.max(dim=0).values+margin]).to(device)
        bounding_box = rend_util.get_2dmask_from_bbox(bbox, intrinsics[0], c2w[0], H, W)
        # mask = model_input['object_mask'].to(device)
        # bounding_box = model_input['bbox_mask'].to(device)
        self.model.mesh.update_vertices(vertices)
        if args.model.input_type in ['tframe','directproj','dispproj']:
            self.model.tposeInfo = {key: val[0].to(device) for key, val in model_input['tposeInfo'].items()}
        elif args.model.input_type == 'invskin':
            self.model.transInfo = {key: val[0].to(device) for key, val in model_input['transformInfo'].items()}

        #self.model.transInfo = {key: val[0].to(device) for key, val in model_input['transformInfo'].items()}
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=args.data.N_rays,
            jittered=args.training.jittered if args.training.get('jittered') is not None else False,
            mask=bounding_box.reshape([-1, H, W])[0] if args.training.get('sample_maskonly',
                                                                          False) is not False else None  # ,
            #todo. Note sampling inside bounding box would yield repeated density update along with outermost side of the box
            # - Need to spread it how?
        )
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
        rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True,
                                             cbfeat_map=cbuvmap,
                                             idfeat_map=idfeatmap,
                                             smpl_param=smplparams,
                                             bounding_box=bbox.to(device)
                                             if args.training.get('sample_maskonly', False) is not False else None,
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
        sample_range = extras.pop('sample_range').reshape((-1, 2, 3))
        range_min = sample_range[:, 0, :].min(0)[0]
        range_max = sample_range[:, 1, :].max(0)[0]
        extras['scalars'].update(
            {'minx': range_min[0], 'miny': range_min[1], 'minz': range_min[2]})
        extras['scalars'].update(
            {'maxx': range_max[0], 'maxy': range_max[1], 'maxz': range_max[2]})

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])

    def val(self,
            args,
            val_in,
            val_gt,
            render_kwargs_test: dict,
            device='cuda'):

        intrinsics = val_in["intrinsics"].to(device)
        c2w = val_in['c2w'].to(device)
        cbuvmap = val_in['cbuvmap'].to(device)
        subject_id = val_in['subject_id'].item() if type(val_in['subject_id']) == torch.Tensor else val_in['subject_id']
        idfeatmap = render_kwargs_test['idfeat_map_all'][subject_id][None].to(device) \
            if render_kwargs_test['idfeat_map_all'] != None else None
        smplparams = val_in['smpl_params']['poses'][:, 0, 3:].to(device) if args.data.smpl_feat != 'none' else None
        vertices = val_in['vertices'][0].to(device)
        self.model.mesh.update_vertices(vertices)
        # self.model.tposeInfo = {key: val[0].to(device) for key, val in val_in['tposeInfo'].items()}
        if args.model.input_type in ['tframe','directproj','dispproj']:
            self.model.tposeInfo = {key: val[0].to(device) for key, val in val_in['tposeInfo'].items()}
        elif args.model.input_type == 'invskin':
            self.model.transInfo = {key: val[0].to(device) for key, val in val_in['transformInfo'].items()}
        B, H, W = val_in['object_mask'].shape
        mask = val_in['object_mask'].reshape(B, -1).to(device)
        # N_rays=-1 for rendering full image
        # todo bbox based sampling to make volume rendering fast
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=-1)
        target_rgb = val_gt['rgb'].to(device)
        target_segm = val_gt['segm'].to(device)
        rgb, depth_v, ret = self.renderer(rays_o, rays_d, detailed_output=True,
                                          calc_normal=True,
                                          cbfeat_map=cbuvmap,
                                          idfeat_map=idfeatmap,
                                          smpl_param=smplparams,
                                          **render_kwargs_test)

        to_img = functools.partial(
            rend_util.lin2img,
            H=H, W=W,
            batched=render_kwargs_test['batched'])
        val_imgs = dict()
        val_imgs['val/gt_rgb'] = to_img(target_rgb)
        if 'label_map_color' in ret:
            val_imgs['val/gt_segm'] = to_img(self.model.decoder.labels_cmap[target_segm])
        val_imgs['val/predicted_rgb'] = to_img(rgb)
        #todo rearrange code!!
        # val_imgs['scalar/psnr'] = torchmetrics.functional.peak_signal_noise_ratio(val_imgs['val/gt_rgb'],
        #                                                                           val_imgs['val/predicted_rgb'], data_range=1.0)
        # val_imgs['scalar/ssim'] = torchmetrics.functional.structural_similarity_index_measure(val_imgs['val/gt_rgb'].view(1, 3, 256, 256),
        #                                                             val_imgs['val/predicted_rgb'].view(1, 3, 256, 256), data_range=1.0)
        # val_imgs['scalar/mse'] =  torchmetrics.functional.mean_squared_error(val_imgs['val/gt_rgb'], val_imgs['val/predicted_rgb'])
        val_imgs['val/pred_depth_volume'] = to_img((depth_v / (depth_v.max() + 1e-10)).unsqueeze(-1))
        val_imgs['val/pred_mask_volume'] = to_img(ret['mask_volume'].unsqueeze(-1))
        # val_imgs['scalar/mask_iou'] = torchmetrics.functional.jaccard_index(???, val_imgs['val/pred_mask_volume'], num_classes=???)
        if 'depth_surface' in ret:
            val_imgs['val/pred_depth_surface'] = to_img(
                (ret['depth_surface'] / ret['depth_surface'].max()).unsqueeze(-1))
        if 'mask_surface' in ret:
            val_imgs['val/predicted_mask'] = to_img(ret['mask_surface'].unsqueeze(-1).float())
        if 'label_map_color' in ret:
            val_imgs['val/predicted_segm'] = to_img(ret['label_map_color'])
            # val_imgs['scalar/mask_iou'] = torchmetrics.functional.jaccard_index(???, ???, num_classes=???)
        if 'normals_volume' in ret:
            val_imgs['val/predicted_normals'] = to_img(ret['normals_volume'] / 2. + 0.5)

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
        if args.model.input_type in ['tframe','directproj','dispproj']:
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
            it=1, num_iters=5000, lr=1.0e-4, batch_points=10000,
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
                    if args.model.input_type in ['tframe','directproj','dispproj']:
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

                    sdf_on, nablas_on, _, _ = self.model.forward_sdf_with_nablas(pts_on.unsqueeze(0), cbfeat=cbfeat,
                                                                                 idGfeat=idGfeat,
                                                                                 idCfeat=idCfeat,
                                                                                 idSfeat=idSfeat,
                                                                                 smpl_param=smpl_param)
                    _, nablas_out, _, _ = self.model.forward_sdf_with_nablas(pts_out.unsqueeze(0), cbfeat=cbfeat,
                                                                             idGfeat=idGfeat, idCfeat=idCfeat, idSfeat=idSfeat,
                                                                             smpl_param=smpl_param)

                    # manifold loss
                    loss_ptson = (sdf_on.abs()).mean()

                    # eikonal loss
                    loss_eik = 1 * ((nablas_out.norm(2, dim=-1) - 1) ** 2).mean()

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
        #todo make a code that do not using segnet or separate idC feat with idSfeat
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
    render_kwargs_test['perturb'] = False

    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])

    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer

