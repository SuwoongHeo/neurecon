#!/usr/bin/env bash
#config_path="./config_test/unisurf.yaml"
#config_path="./config_test/volsdf_siren.yaml"
#config_path="./config_test/neus_nomask.yaml"
#config_path="./config_test/neus.yaml"
#config_path="./config_test/nerfpp.yaml"
#config_path="./config_test/neusseg_nomask.yaml"
config_path="./config_test/neusseg_mask.yaml"
gpu_ids="0"
num_gpus=1
master_port=12322
echo current path is $(pwd)
CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=${num_gpus} --nnodes=1 train.py --ddp --config ${config_path}