import os
import torch

from models.frameworks import get_model
from models.base import get_optimizer, get_scheduler

from utils.checkpoints import CheckpointIO
from utils import rend_util, train_util, mesh_util, io_util
from utils.dist_util import get_local_rank, init_env, is_master, get_rank, get_world_size
from dataio import get_data

def main_function(args):
    args.update({'device_ids': [0]}) #Temporary
    init_env(args)
    local_rank = get_local_rank()
    exp_dir = args.training.exp_dir
    mesh_dir = os.path.join(exp_dir, 'meshes')
    device = torch.device('cuda', local_rank)

    dataset, val_dataset = get_data(args, return_val=True, val_downscale=args.data.get('val_downscale', 4.0))
    # Create model
    model, trainer, render_kwargs_train, render_kwargs_test, volume_render_fn = get_model(args)
    model.to(device)

    render_kwargs_train['H'] = dataset.H
    render_kwargs_train['W'] = dataset.W
    render_kwargs_test['H'] = val_dataset.H
    render_kwargs_test['W'] = val_dataset.W

    # build optimizer
    optimizer = get_optimizer(args, model)

    # checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'), allow_mkdir=is_master())
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
        optimizer=optimizer,
    )

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.ckpt_file,
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys,
        map_location=device)
    it = load_dict.get('global_step', 0)

    with torch.no_grad():
        io_util.cond_mkdir(mesh_dir)
        verts, faces = mesh_util.extract_mesh(
            model.implicit_surface,
            filepath=os.path.join('./foo', '{:08d}.ply'.format(it)),
            volume_size=args.data.get('volume_size', 4.0),
            show_progress=is_master())

    # from tools.vis_camera import visualize
    # import numpy as np
    # camera_matrix = next(iter(val_dataset))[1]['intrinsics'].data.cpu().numpy()
    # c2w_ = np.array([ds.cpu().numpy() for ds in val_dataset.c2w_all])
    # visualize(camera_matrix, np.linalg.inv(c2w_), cam_width=0.2, cam_height=0.1, scale_focal=100)

    #todo, modify normalization location into get_rays (not for each ones)
    for viewid in range(len(dataset)):
        (ind, data, gt) = val_dataset.__getitem__(viewid)
        r_o, r_d, s_i = rend_util.get_rays(
            data['c2w'].to(device).unsqueeze(0), data['intrinsics'].to(device).unsqueeze(0),
            render_kwargs_test['H'], render_kwargs_test['W'], N_rays=-1)

    (ind, data, gt) = val_dataset.__getitem__(0)
    r_o, r_d, s_i = rend_util.get_rays(
        data['c2w'].to(device).unsqueeze(0), data['intrinsics'].to(device).unsqueeze(0),
        render_kwargs_test['H'], render_kwargs_test['W'], N_rays=-1)
    # near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
    # rgb, depth_v, ret = volume_render_fn(r_o, r_d, calc_normal=True, detailed_output=True,
    #                                      **render_kwargs_test)

    foo = 1

if __name__ == "__main__":
    import argparse
    # Arguments
    parser = argparse.ArgumentParser()
    # standard configs
    # parser.add_argument('--config', type=str, default='config_test/unisurf_debug.yaml', help='Path to config file.')
    parser.add_argument('--config', type=str, default='logs/neus_nomask_jungwoo_exp_a_4/config.yaml', help='Path to config file.')
    parser.add_argument('--resume_dir', type=str, default=None, help='Directory of experiment to load.')
    parser.add_argument("--ddp", action='store_true', help='whether to use DDP to train.')
    parser.add_argument("--port", type=int, default=None, help='master port for multi processing. (if used)')
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    main_function(config)