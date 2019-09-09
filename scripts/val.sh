#!/bin/sh


if [ "$#" -eq 0 ]; then
    echo "1. trial (xp name)"
    exit 0
fi

if [ "$#" -ne 1 ]; then
    echo "Error: bad number of arguments"
    echo "1. trial (xp name)"
    exit 1
fi

trial="$1"

log_dir=res/"$trial"


python3 -m train.val_pspnet \
  --trial "$trial" \
  --trained_net pth/from-paper/CMU-CS-Vistas-CE.pth \
  --data_id 1 \
  --img_root_dir meta/trunks/dataset/img_dir \
  --seg_root_dir meta/trunks/dataset/seg_dir \
  --val_crop_size 384 \
  --train_crop_size 384 \
  --stride_rate 0.66 \
  --n_workers 0   
