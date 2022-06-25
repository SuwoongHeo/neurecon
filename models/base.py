from utils.print_fn import log
from utils.logger import Logger

import functools
import math
import numbers
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch import autograd
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class Sine(nn.Module):
    def __init__(self, w0, FiLMLayer=False):
        super().__init__()
        self.w0 = w0

        #inspired by pi-GAN below,
        # NOTE(from swheo) : pi-GAN used mapping network to get scale(frequency),shift from mapping network having input latent z
        # https://github.com/marcoamonteiro/pi-GAN/
        self.FiLMLayer = FiLMLayer
        if FiLMLayer:
            self.scale = nn.Parameter(data=torch.Tensor([1.]), requires_grad=True)
            self.shift = nn.Parameter(data=torch.Tensor([0.]), requires_grad=True)

    def forward(self, x):
        #todo Filmed siren as,
        return torch.sin(self.scale*self.w0*x + self.shift) if self.FiLMLayer else torch.sin(self.w0 * x)


class SirenLayer(nn.Linear):
    def __init__(self, input_dim, out_dim, *args, is_first=False, **kwargs):
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = 30
        self.c = 6
        super().__init__(input_dim, out_dim, *args, **kwargs)
        self.activation = Sine(self.w0)

    # override
    def reset_parameters(self) -> None:
        # NOTE: in offical SIREN, first run linear's original initialization, then run custom SIREN init.
        #       hence the bias is initalized in super()'s reset_parameters()
        # Swheo; Checked, same as original weights
        super().reset_parameters()
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.weight.uniform_(-w_std, w_std)

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=nn.ReLU(inplace=True), **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        if self.activation is not None:
            out = self.activation(out)
        return out

class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim,
                D = 4, W = 256, skips = [], W_up = [],
                featcats=[], featdim=0,
                activation_intmed = None,
                activation_output = None,
                weight_init = None,
                weight_norm = True,
                use_siren = False):
        super().__init__()
        self.input_dim = in_dim
        self.out_dim = out_dim
        self.D = D
        self.W = W
        self.skips = np.array(skips)
        self.W_up = np.array(W_up)
        self.featcats = np.array(featcats)

        outdims = np.array([W] * (D + 1))
        outdims[D] = out_dim

        indims = np.array([W] * (D + 1))
        indims[0] = in_dim

        if len(W_up) > 0:
            outdims[self.W_up] = outdims[self.W_up] * 2
            indims[self.W_up - 1] = indims[self.W_up - 1] * 2

        if len(skips) > 0:
            indims[self.skips] = indims[self.skips] + in_dim # or # W
            outdims[self.skips - 1] = indims[self.skips-1] # or # W - in_dim

        if len(featcats) > 0 and featdim != 0:
            indims[self.featcats] = indims[self.featcats] + featdim

        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        layers = []
        for l in range(D + 1):
            if l != D:
                layer = SirenLayer(indims[l], outdims[l], is_first=(l == 0)) if use_siren \
                    else DenseLayer(indims[l], outdims[l], activation=activation_intmed)
            else:
                layer = DenseLayer(indims[l], outdims[l], activation=activation_output)

            if weight_norm and weight_init is None:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        if weight_init is not None:
            weight_init(layers)
            if weight_norm:
                layers = [nn.utils.weight_norm(layer_) for layer_ in layers]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, infeat=None):
        h = x
        for i in range(self.D):
            if i in self.skips:
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            if i in self.featcats and infeat is not None:
                h = torch.cat([h, infeat], dim=-1) / np.sqrt(2)

            h = self.layers[i](h)

        out = self.layers[-1](h)

        return out

def weight_init_SAL(layers, skips, radius_init, embed_multires):
    D = len(layers) - 1
    input_ch = layers[0].in_features
    for l, m in enumerate(layers):
        # --------------
        # sphere init, as in SAL / IDR.
        # --------------
        if l == D:
            nn.init.normal_(m.weight, mean=np.sqrt(np.pi) / np.sqrt(m.in_features), std=0.0001)
            nn.init.constant_(m.bias, -radius_init)
        elif embed_multires > 0 and l == 0:
            torch.nn.init.constant_(m.bias, 0.0)
            torch.nn.init.constant_(m.weight[:, 3:], 0.0)  # let the initial weights for octaves to be 0.
            torch.nn.init.normal_(m.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(m.out_features))
        elif embed_multires > 0 and l in skips:
            torch.nn.init.constant_(m.bias, 0.0)
            torch.nn.init.normal_(m.weight, 0.0, np.sqrt(2) / np.sqrt(m.out_features))
            torch.nn.init.constant_(m.weight[:, -(input_ch - 3):],
                                    0.0)  # NOTE: this contrains the concat order to be  [h, x_embed]
        else:
            nn.init.constant_(m.bias, 0.0)
            nn.init.normal_(m.weight, 0.0, np.sqrt(2) / np.sqrt(m.out_features))

#Todo rename this shitty name
class ImplicitSurface(nn.Module):
    def __init__(self,
                 W=256, D=8, W_geo_feat=256, input_dim=3, input_feat=0,  output_dim=1, radius_init=1.0, obj_bounding_size=2.0, embed_multires=6,
                 skips=[4], featcats=[], W_up=[], geometric_init=True, weight_norm=True, use_siren=False
                 ):
        """
        W_geo_feat: to set whether to use nerf-like geometry feature or IDR-like geometry feature.
            set to -1: nerf-like, the output feature is the second to last level's feature of the geometry network.
            set to >0: IDR-like ,the output feature is the last part of the geometry network's output.
        """
        super().__init__()
        self.radius_init = radius_init
        self.register_buffer('obj_bounding_size', torch.tensor([obj_bounding_size]).float())
        self.geometric_init = geometric_init
        self.D = D
        self.W = W
        self.W_geo_feat = W_geo_feat
        self.output_dim = output_dim
        if use_siren:
            # assert len(skips) == 0, "do not use skips for siren" #todo : get back if it is not working
            self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool))
        self.skips = skips
        self.featcats = featcats
        self.use_siren = use_siren
        self.embed_fn, input_ch = get_embedder(embed_multires, input_dim)
        self.input_feat = input_feat
        weight_init = functools.partial(weight_init_SAL, skips=skips, radius_init=radius_init,
                                        embed_multires=embed_multires)
        input_ch = input_ch + input_feat if len(featcats)==0 else input_ch
        self.mlpnet = MLPNet(in_dim=input_ch, out_dim=self.output_dim + W_geo_feat if W_geo_feat > 0 else self.output_dim,
                             D=D, W=W, skips=skips, W_up=W_up,
                             featcats=featcats, featdim=self.input_feat,
                             activation_intmed=nn.Softplus(beta=100),
                             activation_output=None,
                             weight_init=weight_init,
                             weight_norm=weight_norm, use_siren=use_siren)


    # siren_sdf only for now...
    def pretrain_hook(self, configs={}):
        configs['target_radius'] = self.radius_init
        # TODO: more flexible, bbox-like scene bound.
        configs['obj_bounding_size'] = self.obj_bounding_size.item()
        if self.geometric_init and self.use_siren and not self.is_pretrained:
            pretrain_siren_sdf(self, **configs)
            self.is_pretrained = ~self.is_pretrained
            return True
        return False

    def forward(self, x: torch.Tensor, return_h = False):
        feat = None
        if self.input_feat>0:
            if len(self.featcats)>0:
                feat = x[..., :self.input_feat]
                x = self.embed_fn(x[..., self.input_feat:])
            else:
                x = torch.cat([x[..., :self.input_feat], self.embed_fn(x[..., self.input_feat:])], dim=-1)
        else:
            x = self.embed_fn(x)

        out = self.mlpnet(x, feat)

        if self.W_geo_feat > 0:
            h = out[..., self.output_dim:]
            out = out[..., :self.output_dim].squeeze(-1)
        else:
            out = out.squeeze(-1)
        if return_h:
            return out, h
        else:
            return out

    # Computing local gradient (normal)
    def forward_with_nablas(self,  x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            implicit_surface_val, h = self.forward(x, return_h=True)
            nabla = autograd.grad(
                implicit_surface_val,
                x,
                torch.ones_like(implicit_surface_val, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]
        if not has_grad:
            implicit_surface_val = implicit_surface_val.detach()
            nabla = nabla.detach()
            h = h.detach()
        return implicit_surface_val, nabla, h

def pretrain_siren_sdf(
    implicit_surface: ImplicitSurface,
    num_iters=5000, lr=1.0e-4, batch_points=5000, 
    target_radius=0.5, obj_bounding_size=3.0,
    logger=None):
    #--------------
    # pretrain SIREN-sdf to be a sphere, as in SIREN and Neural Lumigraph Rendering
    #--------------
    from tqdm import tqdm
    from torch import optim
    device = next(implicit_surface.parameters()).device
    optimizer = optim.Adam(implicit_surface.parameters(), lr=lr)
    
    with torch.enable_grad():
        for it in tqdm(range(num_iters), desc="=> pretraining SIREN..."):
            pts = torch.empty([batch_points, 3]).uniform_(-obj_bounding_size, obj_bounding_size).float().to(device)
            sdf_gt = pts.norm(dim=-1) - target_radius
            # sdf_pred = implicit_surface.forward(pts)
            sdf_pred, nablas_pred, _ = implicit_surface.forward_with_nablas(pts)

            loss = F.l1_loss(sdf_pred, sdf_gt, reduction='mean')
            # [B, N_rays, N_pts]
            nablas_norm = torch.norm(nablas_pred, dim=-1)
            loss += 0.1 * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if logger is not None:
                logger.add('pretrain_siren', 'loss_l1', loss.item(), it)

class RadianceNet(nn.Module):
    def __init__(self,
        W=256, D=4, W_geo_feat=256, input_dim_pts=3, input_feat=0,
        embed_multires=6, embed_multires_view=4, output_dim=3,
        skips=[], featcats=[], W_up=[],
        use_view_dirs=True,
        weight_norm=True,
        use_siren=False,):
        super().__init__()

        input_dim_views = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.input_feat = input_feat
        self.featcats = featcats
        self.skips = skips
        self.D = D
        self.W = W
        self.use_view_dirs = use_view_dirs
        self.embed_fn, input_ch_pts = get_embedder(embed_multires, input_dim=input_dim_pts)
        if use_view_dirs:
            self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view, input_dim=input_dim_views)
            in_dim_0 = input_ch_pts + input_ch_views + 3 + W_geo_feat #3 for normal
        else:
            in_dim_0 = input_ch_pts + W_geo_feat

        input_ch = in_dim_0 + input_feat if len(featcats)==0 else in_dim_0
        self.mlpnet = MLPNet(in_dim=input_ch, out_dim=output_dim,
                             D=D, W=W, skips=skips, W_up=W_up,
                             featcats=featcats, featdim=input_feat,
                             activation_intmed=nn.ReLU(inplace=True),
                             activation_output=None if output_dim>3 else nn.Sigmoid(),
                             weight_init=None,
                             weight_norm=weight_norm, use_siren=use_siren)

    def forward(
        self, 
        x: torch.Tensor, 
        view_dirs: torch.Tensor, 
        normals: torch.Tensor, 
        geometry_feature: torch.Tensor):
        feat = None
        if self.input_feat>0:
            if len(self.featcats)>0:
                feat = x[..., :self.input_feat]
                x = self.embed_fn(x[..., self.input_feat:])
            else:
                x[..., self.input_feat:] = self.embed_fn(x[..., self.input_feat:])
        else:
            x = self.embed_fn(x)
        # calculate radiance field
        if self.use_view_dirs:
            view_dirs = self.embed_fn_view(view_dirs)
            radiance_input = torch.cat([x, view_dirs, normals, geometry_feature], dim=-1)
        else:
            radiance_input = torch.cat([x, geometry_feature], dim=-1)

        h = self.mlpnet(radiance_input, feat)

        return h


# modified from https://github.com/yenchenlin/nerf-pytorch
# swheo: modified using https://github.com/Harry-Zhi/semantic_nerf
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_view=3, multires=-1, multires_view=-1, output_ch=4, skips=[4], use_view_dirs=False,
                 enable_semantic=False, num_semantic_classes=0):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.use_view_dirs = use_view_dirs

        self.embed_fn, input_ch = get_embedder(multires, input_dim=input_ch)
        self.embed_fn_view, input_ch_view = get_embedder(multires_view, input_dim=input_ch_view)
        self.enable_semantic=enable_semantic

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_view_dirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            if enable_semantic:
                self.semantic_linear = nn.Sequential(
                    nn.Sequential(nn.Linear(W,W//2), nn.ReLU(W//2))
                    , nn.Linear(W//2,num_semantic_classes)
                )
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        input_pts = self.embed_fn(input_pts)
        input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu_(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        if self.use_view_dirs:
            sigma = self.alpha_linear(h)
            if self.enable_semantic:
                segm_logits = self.semantic_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], dim=-1)

            for i in range(len(self.views_linears)):
                h = self.views_linears[i](h)
                h = F.relu_(h)

            rgb = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)
            segm_logits = None
            rgb = outputs[..., :3]
            sigma = outputs[..., 3:]

        rgb = torch.sigmoid(rgb)
        return (sigma.squeeze(-1), rgb, segm_logits) if self.enable_semantic else (sigma.squeeze(-1), rgb)

class NerfppNetwithAutoExpo(nn.Module):
    def __init__(self,
                 input_ch=3,
                 optim_autoexpo=False,
                 embed_multires = 10,
                 embed_multires_view = 4,
                 use_view_dirs = True,
                 D = 8,
                 W = 256,
                 skips = [4],
                 num_images = -1,
                 ):
        super().__init__()

        self.fg_net = NeRF(input_ch=input_ch, D=D, W=W, skips=skips, multires=embed_multires, multires_view=embed_multires_view, use_view_dirs=use_view_dirs)
        self.bg_net = NeRF(input_ch=input_ch + 1, D=D, W=W, skips=skips, multires=embed_multires, multires_view=embed_multires_view, use_view_dirs=use_view_dirs)
        self.optim_autoexpo = optim_autoexpo
        # todo : autoexpo from https://github.com/Kai-46/nerfplusplus, is currently not used here
        if self.optim_autoexpo:
            assert (num_images<0)
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(str(x), nn.Parameter(torch.Tensor([0.5, 0.]))) for x in range(num_images)]))

    def forward(self, x_fg: torch.Tensor, x_bg: torch.Tensor, view_dirs, indices=-1):
        fg_sigma, fg_radiance = self.fg_net.forward(x_fg, view_dirs)
        bg_sigma, bg_radiance = self.bg_net.forward(x_bg, view_dirs)
        if self.optim_autoexpo:
            autoexpo = self.autoexpo_params[str(indices.numpy()[0])]
            scale = torch.abs(autoexpo[0]) + 0.5
            shift = autoexpo[1]
        else:
            scale = torch.Tensor([1.]).repeat(fg_sigma.shape).to(fg_sigma.device)
            shift = torch.Tensor([0.]).repeat(fg_sigma.shape).to(fg_sigma.device)
            # scale = torch.Tensor([1.]).unsqueeze(-1).to(fg_sigma.device)
            # shift = torch.Tensor([0.]).unsqueeze(-1).to(fg_sigma.device)

        return fg_sigma, fg_radiance, bg_sigma, bg_radiance, scale, shift

class ScalarField(nn.Module):
    # TODO: should re-use some feature/parameters from implicit-surface net.
    def __init__(self, input_ch=3, W=128, D=4, skips=[], init_val=-2.0):
        super().__init__()
        self.skips = skips
        
        pts_linears = [nn.Linear(input_ch, W)] + \
            [nn.Linear(W, W) if i not in skips 
             else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        for linear in pts_linears:
            nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)


        self.pts_linears = nn.ModuleList(pts_linears)
        self.output_linear = nn.Linear(W, 1)
        nn.init.zeros_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, init_val)

    def forward(self, x: torch.Tensor):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu_(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
        out = self.output_linear(h).squeeze(-1)
        return out


def get_optimizer(args, model):    
    if isinstance(args.training.lr, numbers.Number):
        optimizer = optim.Adam(model.parameters(), lr=args.training.lr)
    elif isinstance(args.training.lr, dict):
        lr_dict = args.training.lr
        default_lr = lr_dict.pop('default')
        
        param_groups = []
        select_params_names = []
        for name, lr in lr_dict.items():
            if name in model._parameters.keys():
                select_params_names.append(name)
                param_groups.append({
                    'params': getattr(model, name),
                    'lr': lr
                })
            elif name in model._modules.keys():
                select_params_names.extend(["{}.{}".format(name, param_name) for param_name, _ in getattr(model, name).named_parameters()])
                param_groups.append({
                    'params': getattr(model, name).parameters(),
                    'lr': lr
                })
            else:
                raise RuntimeError('wrong lr key:', name)

        # NOTE: parameters() is just calling named_parameters without returning name.
        other_params = [param for name, param in model.named_parameters() if name not in select_params_names]
        param_groups.insert(0, {
            'params': other_params,
            'lr': default_lr
        })
        
        optimizer = optim.Adam(params=param_groups, lr=default_lr)
    else:
        raise NotImplementedError
    return optimizer


def get_optimizer_(params):
    optimizer = optim.Adam(params)

    return optimizer

def CosineAnnealWarmUpSchedulerLambda(total_steps, warmup_steps, min_factor=0.1):
    assert 0 <= min_factor < 1
    def lambda_fn(epoch):
        """
        modified from https://github.com/Totoro97/NeuS/blob/main/exp_runner.py
        """
        if epoch < warmup_steps:
            learning_factor = epoch / warmup_steps
        else:
            learning_factor = (np.cos(np.pi * ((epoch - warmup_steps) / (total_steps - warmup_steps))) + 1.0) * 0.5 * (1-min_factor) + min_factor
        return learning_factor
    return lambda_fn


def ExponentialSchedulerLambda(total_steps, min_factor=0.1):
    assert 0 <= min_factor < 1
    def lambda_fn(epoch):
        t = np.clip(epoch / total_steps, 0, 1)
        learning_factor = np.exp(t * np.log(min_factor))
        return learning_factor
    return lambda_fn


def get_scheduler(args, optimizer, last_epoch=-1):
    stype = args.training.scheduler.type
    if stype == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, 
                args.training.scheduler.milestones, 
                gamma=args.training.scheduler.gamma, 
                last_epoch=last_epoch)
    elif stype == 'warmupcosine':
        # NOTE: this do not support per-parameter lr
        # from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        # scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer, 
        #     args.training.num_iters, 
        #     max_lr=args.training.lr, 
        #     min_lr=args.training.scheduler.setdefault('min_lr', 0.1*args.training.lr), 
        #     warmup_steps=args.training.scheduler.warmup_steps, 
        #     last_epoch=last_epoch)
        # NOTE: support per-parameter lr
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            CosineAnnealWarmUpSchedulerLambda(
                total_steps=args.training.num_iters, 
                warmup_steps=args.training.scheduler.warmup_steps, 
                min_factor=args.training.scheduler.setdefault('min_factor', 0.1)
            ),
            last_epoch=last_epoch)
    elif stype == 'exponential_step':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            ExponentialSchedulerLambda(
                total_steps=args.training.num_iters,
                min_factor=args.training.scheduler.setdefault('min_factor', 0.1)
            )
        )
    else:
        raise NotImplementedError
    return scheduler

def forward_with_nablas(func, x: torch.Tensor, has_grad_bypass:bool = None):
    has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
    # force enabling grad for normal calculation
    with torch.enable_grad():
        # x = x.requires_grad_(True)
        x.requires_grad_(True)
        out = list(func(x))
        nabla = autograd.grad(
            out[0],
            x,
            torch.ones_like(out[0], device=x.device),
            create_graph=has_grad,
            retain_graph=has_grad,
            only_inputs=True)[0]
        out.insert(1, nabla)
    if not has_grad:
        for i in range(len(out)):
            try:
                out[i] = out[i].detach()
            except AttributeError:
                out[i] = list(out[i])
                for j in range(len(out[i])):
                    out[i][j] = out[i][j].detach()
                out[i] = tuple(out[i])

    return tuple(out)

if __name__ == "__main__":
    def test():
        """
        test siren-sdf pretrain
        """
        from utils.print_fn import logger
        from utils.mesh_util import extract_mesh
        from utils.io_util import cond_mkdir
        # NOTE: 1.0e-3, 1000 batch points overfit.
        lr = 1.0e-4
        num_iters = 5000
        batch_points = 5000
        cond_mkdir('./dev_test/pretrain_siren')
        logger = Logger('./dev_test/pretrain_siren', img_dir='./dev_test/pretrain_siren/imgs', monitoring='tensorboard', monitoring_dir='./dev_test/pretrain_siren/events')
        
        siren_sdf = ImplicitSurface(W_geo_feat=256, skips=[], use_siren=True)
        siren_sdf.cuda()
        for n, p in siren_sdf.named_parameters():
            log.info(n, p.data.norm().item())
        #--------- extract mesh @ 0-th iter.
        # extract_mesh(siren_sdf, volume_size=1.0, filepath='./dev_test/pretrain_siren/0.ply')
        
        #--------- normal train and extract mesh finally.
        siren_sdf.pretrain_hook({'logger': logger, 'lr': lr, 'num_iters': num_iters, 'batch_points': batch_points})
        namebase = "lr={:.3g}_bs={}_num={}".format(lr, num_iters, batch_points)
        for n, p in siren_sdf.named_parameters():
            log.info(n, p.data.norm().item())
        extract_mesh(siren_sdf, volume_size=1.0, filepath='./dev_test/pretrain_siren/{}.ply'.format(namebase))
        torch.save(siren_sdf.state_dict(), './dev_test/pretrain_siren/{}.pt'.format(namebase))
    test()