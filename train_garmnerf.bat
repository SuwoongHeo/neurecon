#!/usr/bin/env bash
#config_path="./config_test/nerfpp.yaml"
config_path="./config_test/garmnerf_debug.yaml"
gpu_ids="3,5"
num_gpus=2
master_port=1234
echo current path is $(pwd)
#CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=${num_gpus} --nnodes=1 train_garmnerf.py --ddp --config ${config_path}
CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --master_port=${master_port} --nproc_per_node=${num_gpus} --nnodes=1 --max_restarts=5 train_garmnerf.py --ddp --config ${config_path}