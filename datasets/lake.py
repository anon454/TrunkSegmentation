

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms

import utils.joint_transforms as joint_transforms
import tools
import datasets.test_data as test_data
ignore_label = 255


class Lake(Dataset):
    """
    Image pair dataset used for weak supervision
    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        image_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
    """

    def __init__(self, args, mode):
        
        self.mode = mode
        self.data_id = args.data_id
        self.img_root_dir = args.img_root_dir
        self.seg_root_dir = args.seg_root_dir
        self.mean_std = ([116.779, 103.939, 123.68], [1, 1, 1])
        self.rot_max = args.rot_max
        self.train_crop_size = args.train_crop_size
        self.val_crop_size = args.val_crop_size

        csv_file = 'meta/list/data/%d/%s.txt'%(args.data_id, mode)
        self.data = np.loadtxt(csv_file, dtype=str)
        
        # mapping from my labels to cityscapes (may be useless in the future)
        self.id_to_trainid = {}
        for i in range(33):
            self.id_to_trainid[i] = ignore_label
        self.id_to_trainid[2] = 10 # sky
        self.id_to_trainid[3] = 8 # vegetation

        
        # data augmentation
        self.random_rotate = (args.random_rotate==1)
        self.random_crop = (args.random_crop==1)
        self.random_flip = (args.random_flip==1)
        self.debug = (args.data_debug==1)

        
        # data transforms
        if mode=='train':
            self.sliding_crop = None
        else:
            self.sliding_crop = joint_transforms.SlidingCropImageOnly(
                    self.val_crop_size, args.stride_rate)
        self.normalize = standard_transforms.Normalize(*self.mean_std)
        self.transform_before_sliding = standard_transforms.Resize(1024)
        #self.convert_to_PIL = standard_transforms.ToPILImage()

    def augment(self, img, mask):
        h, w, c = img.shape

        # small rotation 
        if self.random_rotate:
            mask_mask = (255*np.ones(mask.shape)).astype(np.uint8)
            center = ((w-1)/2.0, (h-1)/2.0)
            rot = np.random.randint(self.rot_max)
            M = cv2.getRotationMatrix2D(center, rot, 1)
            img = cv2.warpAffine(img, M, (w,h))
            mask = cv2.warpAffine(mask, M, (w,h))
            mask_mask = cv2.warpAffine(mask_mask, M, (w,h))
            mask[mask_mask==0] = 0

        # do random crop
        if self.random_crop:
            top = np.random.randint(h -  self.train_crop_size)
            bottom = np.minimum(h, top + self.train_crop_size)
            #print('top-botton: %d -> %d'%(top, bottom))
            
            left = np.random.randint(w - self.train_crop_size)
            right = np.minimum(w, left + self.train_crop_size)
            #print('left-right: %d -> %d'%(left, right))
            
            img = img[top:bottom, left:right, :]
            mask = mask[top:bottom, left:right]
            
        # flip horizontally if needed
        if self.random_flip:
            img = np.flip(img,1)
            mask = np.flip(mask,1)
        
        # debug
        if self.debug:
            cv2.imshow('img', img)
            cv2.imshow('mask', mask)
            palette = [[128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [70, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32],
                    [0, 0, 0]]
            palette_bgr = [[ l[2], l[1], l[0]] for l in palette]

            mask_col = test_data.lab2col(mask, palette_bgr)
            overlay = tools.gen_overlay(img, mask)
            cv2.imshow('mask_col', mask_col)
            cv2.imshow('overlay', overlay)
            cv2.waitKey(0)

        return img, mask


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        
        img_path = '%s/%s'%(self.img_root_dir, self.data[idx,0])
        mask_path = '%s/%s'%(self.seg_root_dir, self.data[idx,1])
        #print(img_path)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
       
        img, mask = self.augment(img, mask)
        img = img.transpose((2,0,1)) # h,w,c -> c,h,w
        #print('img.shape', img.shape)
        img = torch.Tensor(img.astype(np.float32))
        #print('mask.shape', mask.shape)

        # remap label ids to be coherent with cityscapes
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = mask_copy.astype(np.int32)
        mask = torch.from_numpy(mask).long()
        
        # val
        if self.mode=='val':
            
            img = standard_transforms.ToPILImage()(img)
            img = self.transform_before_sliding(img)
            #img = standard_transforms.ToTensor()(img)
            #img = torch.Tensor(np.asarray(img))
            img_slices, slices_info = self.sliding_crop(img)
            img_slices = [self.normalize(standard_transforms.ToTensor()(e)) for e in img_slices]
            img = torch.stack(img_slices, 0)
            return img, mask, torch.LongTensor(slices_info)

        # train
        else:
            img = self.normalize(img)
            return img, mask


