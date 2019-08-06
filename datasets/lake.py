

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset

from datasets.cityscapes import remap_mask
import datasets.cityscapes as cityscapes

ignore_label = cityscapes.ignore_label

class Lake(Dataset):
    """
    Image pair dataset used for weak supervision
    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        image_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
    """

    def __init__(self, mode, data_id, img_root_dir, seg_root_dir,
            joint_transform=None, sliding_crop=None, transform=None,
            target_transform=None, transform_before_sliding=None):

        self.data_id = data_id
        self.img_root_dir = img_root_dir
        self.seg_root_dir = seg_root_dir

        csv_file = 'meta/list/data/%d/%s.txt'%(data_id, mode)
        self.data = np.loadtxt(csv_file, dtype=str)
        #print(self.data)
        #print(self.data.shape)
        #print(self.data[10,0])
        
        self.id_to_trainid = {}
        for i in range(33):
            self.id_to_trainid[i] = ignore_label
        self.id_to_trainid[23] = 2 # sky
        self.id_to_trainid[21] = 3 # vegetation

        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.transform_before_sliding = transform_before_sliding

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        #print(self.img_root_dir)
        #print(self.data[idx,0])
        #print(idx)
        img_path = '%s/%s'%(self.img_root_dir, self.data[idx,0])
        mask_path = '%s/%s'%(self.seg_root_dir, self.data[idx,1])

        img = cv2.imread(img_path)[:,:700]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)


        img = Image.fromarray(img)
        #img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        
        # remap label ids to be coherent with cityscapes
        #mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))


        # TODO: I don't know why we do this
        if self.joint_transform is not None:
            # from 0,1,2,...,255 to 0,1,2,3,... (to set introduced pixels due
            # to transform to ignore)
            mask = remap_mask(mask, 0)
            img, mask = self.joint_transform(img, mask)
            mask = remap_mask(mask, 1)  # back again

        
        # just transform mask to tensor
        if self.target_transform is not None:
            mask = self.target_transform(mask)


        if self.sliding_crop is not None:
            if self.transform_before_sliding is not None:
                img = self.transform_before_sliding(img)
            img_slices, slices_info = self.sliding_crop(img)
            if self.transform is not None: # substract mean, var=1
                img_slices = [self.transform(e) for e in img_slices]
            img = torch.stack(img_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None: # substract mean, var=1
                img = self.transform(img)
            return img, mask


