#!/usr/bin/env bash

# GMFlow without refinement

# number of gpus for training, please set according to your hardware
# by default use all gpus on a machine
# can be trained on 4x 16GB V100 or 2x 32GB V100 or 2x 40GB A100 gpus
NUM_GPUS=1

# CHECKPOINT_DIR=checkpoints/dsec-gmflow && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main.py \
# --resume pretrained/gmflow_with_refine_kitti-8d3b9786.pth \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --batch_size 10 \
# --val_dataset dsec \
# --stage dsec \
# --num_workers 0 \
# --lr 4e-4 \
# --with_speed_metric \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --val_freq 10000 \
# --save_ckpt_freq 100 \
# --num_steps 10000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

## training locally

CHECKPOINT_DIR=checkpoints/dsec-gmflow && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 1 \
--val_dataset dsec \
--stage dsec \
--num_workers 0 \
--num_time_bins 3 \
--lr 4e-4 \
--num_transformer_layers 2 \
--with_speed_metric \
--val_freq 10000 \
--save_ckpt_freq 100 \
--num_steps 10000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# an example: resume chairs training
# CHECKPOINT_DIR=checkpoints/chairs-gmflow && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume checkpoints/chairs-gmflow/checkpoint_latest.pth \
# --batch_size 16 \
# --val_dataset chairs sintel kitti \
# --lr 4e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 10000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

