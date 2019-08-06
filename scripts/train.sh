#!/bin/sh

trial=0

if ! [ -d res/"$trial" ]; then
  mkdir -p res/"$trial"/log
  mkdir -p res/"$trial"/snap
  mkdir -p res/"$trial"/val
fi

python3 train/train_pspnet.py \
  --trial "$trial" \
  --train_batch_size 1 \
  --lr 2.5e-5 \
  --lr_decay 1 \
  --max_iter 30000 \
  --weight_decay 1e-4 \
  --momentum 0.9 \
  --startnet pth/from-paper/CMU-CS-Vistas-CE.pth \
  --seg_loss_weight 1 \
  --print_freq 10 \
  --val_interval 500 \
  --stride_rate 0.66 \
  --n_workers 1 \
  --data_id 0 \
  --img_root_dir /mnt/lake/ \
  --seg_root_dir /home/gpu_user/assia/ws/datasets/lake/datasets/seg

