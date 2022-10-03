from models.base import ImplicitSurface, NeRF, RadianceNet, forward_with_nablas, get_embedder, DenseLayer
from models.layers.spconvnet import SparseConvNet
from models.trainers.trainer_neuralbody import Trainer
from utils import rend_util, train_util, mesh_util, io_util

from utils.geometry import Mesh, project2closest_face, texcoord2imcoord

import os
import copy
import functools

from spconv import pytorch as spconv

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from kornia.morphology import dilation

import matplotlib.cm as colormap
asset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../assets/"
smpl_dir = '../../assets/smpl/smpl'

class NeuralBody(nn.Module):
    def __init__(self,
                 W_smpl_emb=16,
                 W_spconv_out=128,
                 W_frame_emb=128,
                 num_train_frames=1,
                 decoder_cfg=dict(),
                 **kwargs):
        super().__init__()
        """
        Combinations of Neuralbody and NeuS and SemanticNeRF
        """

        # Canonical body feature module
        self.spconvnet = SparseConvNet(in_dim=W_smpl_emb, out_dim=W_spconv_out)
        # Note. swheo: density_t(x) = model(feature(x, Z, S_t)), color_t(x)=colormodel(feature(x,Z,S_t), pos(d), pos(x), l_t)
        self.density_net = ImplicitSurface(
            W_geo_feat=decoder_cfg['W_geo_feat'], input_dim=decoder_cfg['input_ch'], obj_bounding_size=0.0,
            activation_intmed=nn.ReLU(),
            **decoder_cfg['surface_cfg'])
        # self.latent_fc = nn.Conv1d(decoder_cfg['W_geo_feat']+W_frame_emb, 256, 1)
        self.latent_fc = DenseLayer(decoder_cfg['W_geo_feat']+W_frame_emb, decoder_cfg['W_geo_feat'], activation=None)
        # Note. swheo: latent MLP will process geofeat before input to the radiance net, treat it as feature
        self.radiance_net = RadianceNet(
            W_geo_feat=0, activation_output=nn.Sigmoid(), **decoder_cfg['radiance_cfg'])

        self.mesh = Mesh(file_name=os.path.join(asset_dir, 'smpl/smpl/smpl_uv.obj'))
        self.mesh_latent = nn.Embedding(self.mesh.vertices.shape[0], W_smpl_emb) # SMPL Point features
        self.frame_latent = nn.Embedding(num_train_frames, W_frame_emb)

    def to(self, device):
        new_self = super(NeuralBody, self).to(device)
        new_self.mesh.to(device)

        return new_self

    def encode_sparse_voxel(self, grid_verts, volume_shape, batch_size):
        mesh_latent = self.mesh_latent(torch.arange(0, self.mesh.vertices.shape[0]).to(grid_verts.device))
        xyzc = spconv.SparseConvTensor(mesh_latent, grid_verts, volume_shape, batch_size)
        feature_volume = self.spconvnet(xyzc)

        return feature_volume

    def sample_voxel_feature(self, x, feature_volume, R, Th, bounds, voxel_size, volume_shape):
        if len(x.shape)<3:
            x = x[None]
        if type(volume_shape) != torch.Tensor:
            volume_shape = torch.tensor(volume_shape)
        B, N_pts, N_dim= x.shape
        # pts_to_can_pts
        x_can = (x - Th)@R
        dhw = x_can[..., [2,1,0]]
        dhw = dhw -  bounds[:, 0, [2,1,0]]
        dhw = dhw / torch.tensor(voxel_size).to(dhw)
        # convert the voxel coordinate to [-1,1]
        dhw = dhw / volume_shape.to(dhw) * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        voxel_coords = dhw[..., [2,1,0]][:, None, None]
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,
                                    voxel_coords,
                                    padding_mode='zeros',
                                    align_corners=True)
            features.append(feature.view((B, feature.shape[1], N_pts)).permute((0, 2, 1)))
        features = torch.cat(features, dim=-1)
        # features = features.view(features.size(0), -1, features.size(4))
        voxel_coords = voxel_coords.view((B, N_pts, N_dim))
        return features, voxel_coords


    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor, xyzc_features: torch.Tensor, latent_idx, compute_rgb=False):
        out = self.density_net(xyzc_features, return_h=compute_rgb)
        sigma = out[0] if compute_rgb else out
        if not compute_rgb:
            return sigma
        else:
            latent_in = self.frame_latent(latent_idx[:,0])
            # latent_in = latent_in[...,None].expand(*latent_in.shape, out[1].size(1))
            latent_in = latent_in.expand(*out[1].shape[:2], latent_in.shape[-1])
            l_t = self.latent_fc(torch.cat((out[1], latent_in), dim=-1))
            radiance_in = torch.cat((l_t, x), dim=-1)
            rgb = self.radiance_net(radiance_in, view_dirs, torch.Tensor(0).to(x.device), torch.Tensor(0).to(x.device)) #No use nablas
            return sigma, rgb
        # todo segmentation?

def get_model(args):
    model_config = {
        # todo make adaptive usage for 'use_frame_latent'
        'use_segm' : args.model.setdefault('use_segm',False),
        'W_smpl_emb' : args.model.setdefault('W_smpl_emb', 16),
        'W_spconv_out': args.model.setdefault('W_spconv_out', 128),
        'W_frame_emb' : args.model.setdefault('W_frame_emb', 128),
        'num_train_frames' : args.data.num_frame
    }

    decoder_cfg = {
        'W_geo_feat': args.model.decoder.setdefault('W_geometry_feature', 256),
    }

    decoder_cfg['input_ch'] = 0 # NB doesn't use pts for shape estimation

    surface_cfg = {
        'embed_multires': args.model.decoder.surface.setdefault('embed_multires', -1),
        'D': args.model.decoder.surface.setdefault('D', 3),
        'W': args.model.decoder.surface.setdefault('W', 256),
        'W_up': args.model.decoder.surface.setdefault('W_up', []),
        'skips': args.model.decoder.surface.setdefault('skips', []),
        'featcats': args.model.decoder.surface.setdefault('featcats', []),
        'weight_norm': False,  # Just as NB
        'geometric_init': False, # Just as NB
        'input_feat': 352 # Sparseconv conv1 32 + conv2 64 + conv3 128 + conv4 128
    }
    # latent_fc 256 + xyz_emb (3+3*2*9)63 + view_emb (3+3*2*4)27
    radiance_cfg = {
        'input_dim_pts' : args.model.decoder.radiance.setdefault('input_dim_pts', 3),
        'embed_multires': args.model.decoder.radiance.setdefault('embed_multires', 10), #Just as NB
        'embed_multires_view': args.model.decoder.radiance.setdefault('embed_multires_view', 4),
        'use_view_dirs': args.model.decoder.radiance.setdefault('use_view_dirs', True),
        'use_normals' : args.model.decoder.radiance.setdefault('use_normals', False),
        'weight_norm' : False, #Just as NB
        'D': args.model.decoder.radiance.setdefault('D', 1),
        'W': args.model.decoder.radiance.setdefault('W', 128),
        'W_up': args.model.decoder.radiance.setdefault('W_up', []),
        'skips': args.model.decoder.radiance.setdefault('skips', []),
        'featcats': args.model.decoder.radiance.setdefault('featcats', []),
        'input_feat': decoder_cfg['W_geo_feat']
    }

    if model_config['use_segm']:
        segmentation_cfg = {
            'input_dim_pts': args.model.decoder.segmentation.setdefault('input_dim_pts', 3),
            'embed_multires': args.model.decoder.segmentation.setdefault('embed_multires', 10), #Just as NB
            'embed_multires_view': args.model.decoder.segmentation.setdefault('embed_multires_view', -1),
            'use_view_dirs': args.model.decoder.segmentation.setdefault('use_view_dirs', False),
            'use_normals': args.model.decoder.radiance.setdefault('use_normals', False),
            'weight_norm': False,  # Just as NB
            'D': args.model.decoder.segmentation.setdefault('D', 2),
            'W': args.model.decoder.segmentation.setdefault('W', 128),
            'W_up': args.model.decoder.segmentation.setdefault('W_up', []),
            'featcats': args.model.decoder.segmentation.setdefault('featcats', []),
            'skips': args.model.decoder.segmentation.setdefault('skips', []),
            'output_dim': args.model.decoder.segmentation.setdefault('output_dim', 4),
            'input_feat': decoder_cfg['W_geo_feat']
        }
    else:
        segmentation_cfg = None

    decoder_cfg['surface_cfg'] = surface_cfg
    decoder_cfg['radiance_cfg'] = radiance_cfg
    decoder_cfg['segmentation_cfg'] = segmentation_cfg

    model_config['decoder_cfg'] = decoder_cfg

    model = NeuralBody(**model_config)

    ## render kwargs

    render_kwargs_train = {
        # upsample config
        # [official_solution, direct_more, direct_use]
        'with_mask': args.training.setdefault('with_mask', True),
        # 'obj_bounding_radius': args.model.decoder.setdefault('obj_bounding_radius', 0.0),

        'batched': args.data.batch_size is not None,
        'perturb': args.model.setdefault('perturb', True),  # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False), # NB
        # NB specific
        'voxel_size': args.model.setdefault('voxel_size', [0.005, 0.005, 0.005]),
        'enlarge_box': args.model.setdefault('enlarge_box', 0.005),
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['maskonly'] = args.data.val_rendermaskonly
    render_kwargs_test['perturb'] = False
    render_kwargs_test['mesh_th'] = args.data.setdefault('mesh_th', 50)
    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])

    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer

