#!/bin/sh

trial=0


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
  --max_epoch 2 \
  --batch_size 3 \
  --lr 2.5e-5 \
  --lr_decay 1 \
  --weight_decay 1e-4 \
  --momentum 0.9 \
  --startnet pth/from-paper/CMU-CS-Vistas-CE.pth \
  --log_interval 10 \
  --summary_interval 10 \
  --val_interval 1 \
  --data_id 0 \
  --img_root_dir /mnt/lake/ \
  --seg_root_dir /home/gpu_user/assia/ws/datasets/lake/datasets/seg \
  --val_crop_size 713 \
  --train_crop_size 384 \
  --stride_rate 0.66 \
  --n_workers 1 \
  --random_rotate 1 \
  --rot_max 10 \
  --random_crop 1 \
  --random_flip 1 \
  --data_debug 1
  
