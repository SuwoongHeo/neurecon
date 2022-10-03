from models.frameworks import get_model
from models.base import get_optimizer_, get_scheduler
from utils import rend_util, train_util, mesh_util, io_util
from utils.dist_util import get_local_rank, init_env, is_master, get_rank, get_world_size
from utils.print_fn import log
from utils.logger import Logger
from utils.checkpoints import CheckpointIO
from dataio import get_data

import os
import sys
import time
import functools
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def main_function(args):
    args.update({'device_ids': [2]})  # Temporary
    init_env(args)

    # ----------------------------
    # -------- shortcuts ---------
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    i_backup = int(args.training.i_backup // world_size) if args.training.i_backup > 0 else -1
    i_val = int(args.training.i_val // world_size) if args.training.i_val > 0 else -1
    i_val_mesh = int(args.training.i_val_mesh // world_size) if args.training.i_val_mesh > 0 else -1
    special_i_val_mesh = [int(i // world_size) for i in [3000, 5000, 7000]]
    exp_dir = args.training.exp_dir
    mesh_dir = os.path.join(exp_dir, 'meshes')

    device = torch.device('cuda', local_rank)

    # logger
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, 'imgs'),
        monitoring=args.training.get('monitoring', 'tensorboard'),
        monitoring_dir=os.path.join(exp_dir, 'events'),
        rank=rank, is_master=is_master(), multi_process_logging=(world_size > 1))

    log.info("=> Experiments dir: {}".format(exp_dir))

    if is_master():
        # backup codes
        io_util.backup(os.path.join(exp_dir, 'backup'))

        # save configs
        io_util.save_config(args, os.path.join(exp_dir, 'config.yaml'))

    dataset, val_dataset = get_data(args, return_val=True, val_downscale=args.data.get('val_downscale', 4.0))
    bs = args.data.get('batch_size', None)
    if args.ddp:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=bs,
                                                 num_workers=args.data.get('num_workers', 0))
        val_sampler = DistributedSampler(val_dataset)
        valloader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, batch_size=bs, num_workers=0)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=bs,
                                shuffle=True,
                                num_workers=args.data.get('num_workers', 0),
                                pin_memory=args.data.get('pin_memory', False))
        valloader = DataLoader(val_dataset,
                               batch_size=1,
                               shuffle=True,
                               num_workers=0)

    # Create model
    model, trainer, render_kwargs_train, render_kwargs_test, volume_render_fn = get_model(args)
    model.to(device)
    log.info(model)
    log.info("=> Nerf params: " + str(train_util.count_trainable_parameters(model)))

    # build optimizer
    if 'idfeat_map_all' in render_kwargs_train.keys():
        if render_kwargs_train['idfeat_map_all'] != None:
            optimizer = get_optimizer_([{"params": model.parameters(), "lr": args.training.lr},
                                        {"params": render_kwargs_train['idfeat_map_all'], "lr": args.training.lr_idfeat}
                                        ])
            model_params = {"model":model, "idfeat":render_kwargs_train['idfeat_map_all']}
    else:
        optimizer = get_optimizer_([{"params": model.parameters(), "lr": args.training.lr}])
        model_params = {"model": model}

    # checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'), allow_mkdir=is_master())
    if world_size > 1:
        dist.barrier()
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **model_params,
        optimizer=optimizer,
    )

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.ckpt_file,
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys,
        map_location=device)

    logger.load_stats('stats.p')  # this will be used for plotting
    it = load_dict.get('global_step', 0)
    epoch_idx = load_dict.get('epoch_idx', 0)

    # pretrain if needed. must be after load state_dict, since needs 'is_pretrained' variable to be loaded.
    # ---------------------------------------------
    # -------- init perparation only done in master
    # ---------------------------------------------
    if is_master():
        torch.autograd.set_detect_anomaly(True)
        it_pretrain = load_dict.get('pretrain_step', 0)
        pretrain_config = {'logger': logger, 'device': device, 'it': it_pretrain, 'mesh_dir': mesh_dir}
        if 'lr_pretrain' in args.training and it==0:
            pretrain_config['lr'] = args.training.lr_pretrain
            try:
                # todo tmp
                with torch.no_grad():
                    io_util.cond_mkdir(mesh_dir)
                    _, data_in, _ = dataloader.dataset.__getitem__(0)
                    trainer.val_mesh(args, data_in, os.path.join(mesh_dir, 'pretrained_{:05d}.ply'.format(it_pretrain)),
                                     render_kwargs_test, device)
                isSucesses, it_pretrain = trainer.pretrain(args, dataloader, render_kwargs_test, **pretrain_config)
                if isSucesses:
                    checkpoint_io.save(filename='latest.pt'.format(it), global_step=it, epoch_idx=epoch_idx,
                                       pretrain_step=it_pretrain)
                    with torch.no_grad():
                        io_util.cond_mkdir(mesh_dir)
                        _, data_in, _ = dataloader.dataset.__getitem__(0)
                        trainer.val_mesh(args, data_in, os.path.join(mesh_dir, 'pretrained_{:05d}.ply'.format(it_pretrain)),
                                      render_kwargs_test, device)
            except KeyboardInterrupt:
                checkpoint_io.save(filename='latest.pt'.format(it), global_step=it, epoch_idx=epoch_idx,
                                   pretrain_step=it_pretrain)
                with torch.no_grad():
                    io_util.cond_mkdir(mesh_dir)
                    _, data_in, _ = dataloader.dataset.__getitem__(0)
                    trainer.val_mesh(args, data_in, os.path.join(mesh_dir, 'pretrained_{:05d}.ply'.format(it_pretrain)),
                                     render_kwargs_test, device)
                sys.exit()

    # Parallel training
    if args.ddp:
        trainer = DDP(trainer, device_ids=args.device_ids, output_device=local_rank, find_unused_parameters=False)

    # build scheduler
    scheduler = get_scheduler(args, optimizer, last_epoch=it - 1)
    t0 = time.time()
    if 'idfeat_map_all' in render_kwargs_train.keys():
        if render_kwargs_train['idfeat_map_all'] != None:
            log.info('=> Start training..., it={}, lr={}, lr_feat={}, in {}'.format(it, optimizer.param_groups[0]['lr'],
                                                                                    optimizer.param_groups[1]['lr'],
                                                                                    exp_dir))
    else:
        log.info('=> Start training..., it={}, lr={}, in {}'.format(it, optimizer.param_groups[0]['lr'], exp_dir))
    end = (it >= args.training.num_iters)
    with tqdm(range(args.training.num_iters), disable=not is_master()) as pbar:
        if is_master():
            pbar.update(it)
        while it <= args.training.num_iters and not end:
            try:
                if args.ddp:
                    train_sampler.set_epoch(epoch_idx)
                for (indices, model_input, ground_truth) in dataloader:
                    int_it = int(it // world_size)
                    # -------------------
                    # validate
                    # -------------------
                    if i_val > 0 and int_it % i_val == 0:
                        scalars = {}
                        with torch.no_grad():
                            for idx, (val_ind, val_in, val_gt) in enumerate(valloader):
                                val_imgs = trainer.module.val(args, val_in, val_gt, it, render_kwargs_test, device=device) \
                                    if isinstance(trainer, torch.nn.parallel.distributed.DistributedDataParallel) else \
                                    trainer.val(args, val_in, val_gt, it, render_kwargs_test, device=device)
                                for name, value in val_imgs.items():
                                    if 'val/' in name:
                                        logger.add_imgs(value, name, it, prefix=val_in['tag'][0], monitor=idx == 0)
                                    elif 'scalar/' in name:
                                        if name in scalars.keys():
                                            scalars[name].append(value.data.cpu().numpy().item())
                                        else:
                                            scalars[name] = [value.data.cpu().numpy().item()]
                                for name, value in scalars.items():
                                    logger.add('eval', name, sum(value)/len(value), it)

                    # -------------------
                    # validate mesh
                    # -------------------
                    if is_master():
                        # NOTE: not validating mesh before 3k, as some of the instances of DTU for NeuS training will have
                        # no large enough mesh at the beginning.
                        if i_val_mesh > 0 and (int_it % i_val_mesh == 0 or int_it in special_i_val_mesh) and it != 0:
                            with torch.no_grad():
                                io_util.cond_mkdir(mesh_dir)
                                try:
                                    # todo can we do this in more canonical concept?
                                    (_, val_in, _) = valloader.dataset.__getitem__(0)
                                    trainer.module.val_mesh(args, val_in,
                                                            os.path.join(mesh_dir, '{:08d}.ply'.format(it)),
                                                            render_kwargs_test, device)
                                except AttributeError:
                                    trainer.val_mesh(args, val_in, os.path.join(mesh_dir, '{:08d}.ply'.format(it)),
                                                     render_kwargs_test, device)

                    if it >= args.training.num_iters:
                        end = True
                        break

                    # -------------------
                    # train
                    # -------------------
                    start_time = time.time()
                    ret = trainer.forward(args, indices, model_input, ground_truth, render_kwargs_train, it)

                    losses = ret['losses']
                    extras = ret['extras']

                    for k, v in losses.items():
                        # log.info("{}:{} - > {}".format(k, v.shape, v.mean().shape))
                        losses[k] = torch.mean(v)
                    # torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
                    optimizer.zero_grad()
                    losses['total'].backward()
                    # NOTE: check grad before optimizer.step()
                    if True:
                        grad_norms = train_util.calc_grad_norm(**model_params)
                    optimizer.step()
                    scheduler.step(it)  # NOTE: important! when world_size is not 1

                    # -------------------
                    # logging
                    # -------------------
                    # done every i_save seconds
                    if (args.training.i_save > 0) and (time.time() - t0 > args.training.i_save):
                        if is_master():
                            checkpoint_io.save(filename='latest.pt', global_step=it, epoch_idx=epoch_idx)
                        # this will be used for plotting
                        logger.save_stats('stats.p')
                        t0 = time.time()

                    if is_master():
                        # ----------------------------------------------------------------------------
                        # ------------------- things only done in master -----------------------------
                        # ----------------------------------------------------------------------------
                        lrs = {f'lr_{name}':val['lr'] for name, val in zip(model_params.keys(), optimizer.param_groups)}
                        losses_= {f"loss_{name.replace('loss_','')}":val.item() for name, val in losses.items()}
                        pbar.set_postfix(**grad_norms, **lrs, **losses_)
                        # if render_kwargs_train['idfeat_map_all'] != None:
                        #     pbar.set_postfix(lr=optimizer.param_groups[0]['lr'],
                        #                      lr_feat=optimizer.param_groups[1]['lr'],
                        #                      loss_total=losses['total'].item(), loss_img=losses['loss_img'].item(),
                        #                      loss_eikonal=losses['loss_eikonal'].item(),
                        #                      loss_mask=losses['loss_mask'].item() if 'loss_mask' in losses.keys() else 0.,
                        #                      loss_seg=losses['loss_seg'].item() if 'loss_seg' in losses.keys() else 0.)
                        # else:
                        #     pbar.set_postfix(lr=optimizer.param_groups[0]['lr'],
                        #                      loss_total=losses['total'].item(), loss_img=losses['loss_img'].item(),
                        #                      loss_eikonal=losses['loss_eikonal'].item(),
                        #                      loss_mask=losses['loss_mask'].item() if 'loss_mask' in losses.keys() else 0.,
                        #                      loss_seg=losses['loss_seg'].item() if 'loss_seg' in losses.keys() else 0.)

                        if i_backup > 0 and int_it % i_backup == 0 and it > 0:
                            checkpoint_io.save(filename='{:08d}.pt'.format(it), global_step=it, epoch_idx=epoch_idx)

                    # ----------------------------------------------------------------------------
                    # ------------------- things done in every child process ---------------------------
                    # ----------------------------------------------------------------------------

                    # -------------------
                    # log grads and learning rate
                    for k, v in grad_norms.items():
                        logger.add('grad', k, v, it)
                    logger.add('learning rates', 'whole', optimizer.param_groups[0]['lr'], it)

                    # -------------------
                    # log losses
                    for k, v in losses.items():
                        logger.add('losses', k, v.data.cpu().numpy().item(), it)

                    # -------------------
                    # log extras
                    names = ["radiance", "alpha", "implicit_surface", "implicit_nablas_norm", "sigma_out",
                             "radiance_out"]
                    for n in names:
                        p = "whole"
                        # key = "raw.{}".format(n)
                        key = n
                        if key in extras:
                            logger.add("extras_{}".format(n), "{}.mean".format(p),
                                       extras[key].mean().data.cpu().numpy().item(), it)
                            logger.add("extras_{}".format(n), "{}.min".format(p),
                                       extras[key].min().data.cpu().numpy().item(), it)
                            logger.add("extras_{}".format(n), "{}.max".format(p),
                                       extras[key].max().data.cpu().numpy().item(), it)
                            logger.add("extras_{}".format(n), "{}.norm".format(p),
                                       extras[key].norm().data.cpu().numpy().item(), it)
                    if 'scalars' in extras:
                        for k, v in extras['scalars'].items():
                            logger.add('scalars', k, v.mean().data.cpu().numpy().item(), it)

                    # ---------------------
                    # end of one iteration
                    end_time = time.time()
                    log.debug("=> One iteration time is {:.2f}".format(end_time - start_time))

                    it += world_size
                    if is_master():
                        pbar.update(world_size)
                # ---------------------
                # end of one epoch
                epoch_idx += 1

            except KeyboardInterrupt:
                if is_master():
                    checkpoint_io.save(filename='latest.pt'.format(it), global_step=it, epoch_idx=epoch_idx)
                    # this will be used for plotting
                logger.save_stats('stats.p')
                sys.exit()

    if is_master():
        checkpoint_io.save(filename='final_{:08d}.pt'.format(it), global_step=it, epoch_idx=epoch_idx)
        logger.save_stats('stats.p')
        log.info("Everything done.")


if __name__ == "__main__":
    import argparse

    # Arguments
    parser = argparse.ArgumentParser()
    # standard configs
    # parser.add_argument('--config', type=str, default='config_test/garmnerf_debug_featext.yaml', help='Path to config file.')
    # parser.add_argument('--config', type=str, default='config_test/garmnerf_debug.yaml',
    #                     help='Path to config file.')
    parser.add_argument('--config', type=str, default='config_test/neuralbody_debug.yaml',
                        help='Path to config file.')
    parser.add_argument('--resume_dir', type=str, default=None, help='Directory of experiment to load.')
    parser.add_argument("--ddp", action='store_true', help='whether to use DDP to train.')
    parser.add_argument("--port", type=int, default=None, help='master port for multi processing. (if used)')
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    main_function(config)