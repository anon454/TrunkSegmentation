#!/bin/sh

trial="$1"

python3 -m train.test \
  --trial "$trial" \
  --trained_net $2\
  --data_id 1 \
  --img_root_dir $3\
  --seg_save_root $4\
  --csv_path $5\
  --val_crop_size 384 \
  --train_crop_size 384 \
  --stride_rate 0.66 \
  --n_workers 0   

