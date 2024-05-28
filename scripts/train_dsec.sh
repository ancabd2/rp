#!/usr/bin/env bash

# GMFlow without refinement

# number of gpus for training, please set according to your hardware
# by default use all gpus on a machine
# can be trained on 4x 16GB V100 or 2x 32GB V100 or 2x 40GB A100 gpus
NUM_GPUS=1

# chairs
CHECKPOINT_DIR=checkpoints/dsec-gmflow && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 16 \
--val_dataset dsec \
--stage dsec \
--lr 4e-4 \
--image_size 384 512 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 10000 \
--save_ckpt_freq 10000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# a final note: if your training is terminated unexpectedly, you can resume from the latest checkpoint
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

