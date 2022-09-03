import torch
import numpy as np
from typing import Iterable

def calc_grad_norm(norm_type=2.0, **named_models):
    gradient_norms = {'total': 0.0}
    for name, model in named_models.items():
        if model==None:
            continue
        gradient_norms[name] = 0.0
        if hasattr(model, 'parameters'):
            for p in list(model.parameters()):
                if p.requires_grad and p.grad is not None:
                    param_norm = p.grad.data.norm(norm_type)
                    gradient_norms[name] += param_norm.item() ** norm_type
        elif type(model) == torch.Tensor:
            gradient_norms[name] = model.grad.data.norm(norm_type).item() ** norm_type
        gradient_norms['total'] += gradient_norms[name]
    for k, v in gradient_norms.items():
        gradient_norms[k] = v ** (1.0 / norm_type)
    return gradient_norms

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def batchify_query(query_fn, *args: Iterable[torch.Tensor], chunk, dim_batchify, **kwargs):
    # [(B), N_rays, N_pts, ...] -> [(B), N_rays*N_pts, ...]
    _N_rays = args[0].shape[dim_batchify]
    _N_pts = args[0].shape[dim_batchify+1]
    args = [arg.flatten(dim_batchify, dim_batchify+1) for arg in args]
    _N = args[0].shape[dim_batchify]
    raw_ret = []
    for i in range(0, _N, chunk):
        if dim_batchify == 0:
            args_i = [arg[i:i+chunk] for arg in args]
        elif dim_batchify == 1:
            args_i = [arg[:, i:i+chunk] for arg in args]
        elif dim_batchify == 2:
            args_i = [arg[:, :, i:i+chunk] for arg in args]
        else:
            raise NotImplementedError
        raw_ret_i = query_fn(*args_i, **kwargs)
        if not isinstance(raw_ret_i, tuple):
            raw_ret_i = [raw_ret_i]
        raw_ret.append(raw_ret_i)
    collate_raw_ret = []
    num_entry = 0
    for entry in zip(*raw_ret):
        if isinstance(entry[0], dict):
            tmp_dict = {}
            for list_item in entry:
                for k, v in list_item.items():
                    if k not in tmp_dict:
                        tmp_dict[k] = []
                    tmp_dict[k].append(v)
            for k in tmp_dict.keys():
                # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
                # tmp_dict[k] = torch.cat(tmp_dict[k], dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
                # NOTE: compatible with torch 1.6
                v = torch.cat(tmp_dict[k], dim=dim_batchify)
                tmp_dict[k] = v.reshape([*v.shape[:dim_batchify], _N_rays, _N_pts, *v.shape[dim_batchify+1:]])
            entry = tmp_dict
        if isinstance(entry[0], tuple):
            tmp_list = []
            for entry_item in entry[0]:
                v = torch.cat([entry_item], dim=dim_batchify) #todo check
                tmp_list.append(v.reshape([*v.shape[:dim_batchify], _N_rays, _N_pts, *v.shape[dim_batchify+1:]]))
            entry = tuple(tmp_list)
        else:
            # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
            # entry = torch.cat(entry, dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
            # NOTE: compatible with torch 1.6
            v = torch.cat(entry, dim=dim_batchify)
            entry = v.reshape([*v.shape[:dim_batchify], _N_rays, _N_pts, *v.shape[dim_batchify+1:]])
        collate_raw_ret.append(entry)
        num_entry += 1
    if num_entry == 1:
        return collate_raw_ret[0]
    else:
        return tuple(collate_raw_ret)


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def add_batch_dim(in_dict, dim=0, B=1):
    out = dict()
    for key, val in in_dict.items():
        if isinstance(val, torch.Tensor):
            out[key] = torch.stack([val]*B, dim=dim)
        elif isinstance(val, dict):
            out[key] = add_batch_dim(val, dim=dim, B=B)
        else:
            out[key] = val

    return out