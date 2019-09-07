#!/bin/sh

trial=1

if [ "$#" -eq 0 ]; then
    echo "1. trial (xp name)"
    exit 0
fi

if [ "$#" -ne 1 ]; then
    echo "Error: bad number of arguments"
    echo "1. trial (xp name)"
    exit 1
fi


log_dir=res/"$trial"
if [ -d "$log_dir" ]; then
    while true; do
        read -p ""$log_dir" already exists. Do you want to overwrite it (y/n) ?" yn
        case $yn in
            [Yy]* ) 
                rm -rf "$log_dir"; 
                mkdir -p "$log_dir"/log; 
                mkdir -p "$log_dir"/snap; 
                mkdir -p "$log_dir"/val; 
                break;;
            [Nn]* ) exit;;
            * ) * echo "Please answer yes or no.";;
        esac
    done
else
  mkdir -p "$log_dir"/log
  mkdir -p "$log_dir"/snap
  mkdir -p "$log_dir"/val

fi


python3 -m train.train_pspnet \
  --trial "$trial" \
  --max_epoch 50 \
  --batch_size 3 \
  --lr 2.5e-5 \
  --lr_decay 1 \
  --weight_decay 1e-4 \
  --momentum 0.9 \
  --startnet pth/from-paper/CMU-CS-Vistas-CE.pth \
  --log_interval 50 \
  --summary_interval 10 \
  --val_interval 1000 \
  --data_id 1 \
  --img_root_dir /home/gpu_user/aishwarya/dataset/img_dir \
  --seg_root_dir /home/gpu_user/aishwarya/dataset/seg_dir \
  --val_crop_size 384 \
  --train_crop_size 384 \
  --stride_rate 0.66 \
  --n_workers 0 \
  --random_rotate 0 \
  --rot_max 10 \
  --random_crop 1 \
  --random_flip 1 \
  --data_debug 0
  
