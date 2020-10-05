#!/bin/sh


logdir=$1
trial=$2
if [ -d "$logdir/$trial" ]; then
  rm -rf "$logdir/$trial"; 
  mkdir -p "$logdir/$trial"/log; 
  mkdir -p "$logdir/$trial"/snap; 
  mkdir -p "$logdir/$trial"/val; 
else
  mkdir -p "$logdir/$trial"/log
  mkdir -p "$logdir/$trial"/snap
  mkdir -p "$logdir/$trial"/val
fi


python3 -m train.train_pspnet \
  --trial "$trial" \
  --logdir "$logdir" \
  --csv_path "$3" \
  --seg_root_dir $5\
  --img_root_dir $6\
  --max_epoch 50 \
  --batch_size 3 \
  --lr 2.5e-5 \
  --lr_decay 0 \
  --weight_decay 1e-4 \
  --momentum 0.9 \
  --startnet $4 \
  --log_interval 50 \
  --summary_interval 10 \
  --val_interval 1 \
  --save_interval 10 \
  --data_id 1 \
  --val_crop_size 384 \
  --train_crop_size 384 \
  --stride_rate 0.66 \
  --n_workers 0 \
  --random_rotate 0 \
  --rot_max 10 \
  --random_crop 1 \
  --random_flip 1 \
  --data_debug 0
  
