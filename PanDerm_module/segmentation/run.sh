#!/bin/bash

echo "Running PanDerm experiment on Ham10000"
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --worker 4 --gpu "0,1" --batch_size 1 --test_batch_size 1 \
    --epoch 100 --lr 1e-4 --weight_decay 0.05 \
    --log_name "cae_seg_lr_1e-4_decay_0.05_full" --model "cae_seg" --size 224 \
    --dataset "Ham10000" --dataset_path "/data2/wangzh/datasets/Ham10000" \
    --save_name "/data3/wangzh/experiments/skinfm/finals/cae_seg_ham" \
    --seed 0 --smoke

