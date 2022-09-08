#!/usr/bin/env bash
#config_path="./config_test/nerfpp.yaml"
config_path="./config_test/garmnerf_debug.yaml"
# config_path="./config_test/garmnerf_debug_featext.yaml"
# resume_dir="./logs/garmnerf_masksegm_exp_p_a_disptframe"
gpu_ids="4,5"
num_gpus=2
master_port=1235
echo current path is $(pwd)
# Train
CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --master_port=${master_port} --nproc_per_node=${num_gpus} --nnodes=1 --max_restarts=5 train_garmnerf.py --ddp --config ${config_path}
# Resume
#CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --master_port=${master_port} --nproc_per_node=${num_gpus} --nnodes=1 --max_restarts=5 train_garmnerf.py --ddp --resume_dir ${resume_dir}
