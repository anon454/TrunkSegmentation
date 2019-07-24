

from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes
#from datasets import cityscapes, dataset_configs
#from utils.misc import check_mkdir, get_global_opts, rename_keys_to_match
from utils.misc import rename_keys_to_match

import os
import re
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as standard_transforms
import h5py
import math

import sys
import time

#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# colmap output dir
MACHINE = 0
if MACHINE == 0:
    DATASET_DIR = '/home/abenbihi/ws/datasets/'
    WS_DIR = '/home/abenbihi/ws/'
    EXT_IMG_DIR = '/mnt/data_drive/dataset/Extended-CMU-Seasons/'
    #DATA_DIR = '/mnt/data_drive/dataset/CMU-Seasons/'
elif MACHINE == 1:
    DATASET_DIR = '/home/gpu_user/assia/ws/datasets/'
    WS_DIR = '/home/gpu_user/assia/ws/'
    EXT_IMG_DIR = '/home/gpu_user/assia/ws/datasets/Extended-CMU-Seasons/'
    #DATA_DIR = '/home/abenbihi/ws/datasets/CMU-Seasons/'
else:
    print('Get you MTF MACHINE macro correct !')
    exit(1)

    
META_DIR = '%s/life_saver/datasets/CMU-Seasons/meta/'%WS_DIR

NETWORK_FILE = 'pth/from-paper/CMU-CS-Vistas-CE.pth'
NUM_CLASS = 19

def segment(slice_id, cam_id, survey_id):
    ## Predefined image sets and paths
    #dset = dataset_configs.CmuConfig()
    #args['img_path'] = dset.test_im_folder
    #args['img_ext'] = dset.im_file_ending
    #img_folder = dset.test_im_folder

    # output dir
    save_folder = 'res/%d/%d_%d/'%(slice_id, cam_id, survey_id)
    if not os.path.exists('%s/col'%save_folder):
        os.makedirs('%s/col'%save_folder)
    if not os.path.exists('%s/lab'%save_folder):
        os.makedirs('%s/lab'%save_folder)
    if not os.path.exists('%s/prob'%save_folder):
        os.makedirs('%s/prob'%save_folder)

    for class_id in range(NUM_CLASS):
        if not os.path.exists('%s/prob/class_%d'%(save_folder, class_id)):
            os.makedirs('%s/prob/class_%d'%(save_folder, class_id))
        #if not os.path.exists('%s/lab/class_%d'%(save_folder, class_id))


    # get all file names
    meta_fn = '%s/surveys/%d/fn/c%d_%d.txt'%(META_DIR, slice_id, cam_id,
            survey_id)

    filenames_ims = ['%s/slice%d/query/%s'%(EXT_IMG_DIR, slice_id, l) for l in
            [ll.split("\n")[0] for ll in open(meta_fn, 'r').readlines()]]
    filenames_segs = ['%s/col/%s.png'%(save_folder, l) for l
            in [ll.split("\n")[0].split(".")[0] for ll in open(meta_fn, 'r').readlines()]]
    
    
    #for i, l in enumerate(filenames_ims):
    #    print(l)
    #    input(filenames_segs[i])

    # network model
    print("Loading specified network from %s"%META_DIR)
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network().to(device)
    print('load model ' + NETWORK_FILE)
    state_dict = torch.load(NETWORK_FILE, map_location=lambda storage, 
            loc: storage)
    # needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    net.load_state_dict(state_dict)
    net.eval()


    # data proc
    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform
    # make sure crop size and stride same as during training
    sliding_crop = joint_transforms.SlidingCropImageOnly(
        713, 2/3.)


    # encapsulate pytorch model in Segmentor class
    print("Class number: %d"%net.n_classes) # 19
    segmentor = Segmentor(
            net,
            net.n_classes,
            colorize_fcn = cityscapes.colorize_mask,
            n_slices_per_pass = 10)

    # let's go
    count = 1
    t0 = time.time()
    for im_file, save_path in zip(filenames_ims, filenames_segs):
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
        print(save_path)
        segmentor.run_and_save(
            im_file,
            save_path,
            save_folder,
            pre_sliding_crop_transform = pre_validation_transform,
            sliding_crop = sliding_crop,
            input_transform = input_transform,
            skip_if_seg_exists = True,
            use_gpu = True)
        count += 1
        if count == 3:
            break




if __name__ == '__main__':
    ##global_opts = get_global_opts()

    ##DATA_ROOT_DIR = '/home/abenbihi/ws/datasets/correspondence/'

    #
    ##CMU_DIR = 'CMU/segmented_images/testing/imgs/' 
    #EXT = '.png'

    #network_folder = '' # not used for now because I specify the network file

    #args = {
    #    'use_gpu': True,
    #    # 'miou' (miou over classes present in validation set), 'acc'
    #    'validation_metric': 'miou',
    #    'img_set': 'cmu',  # ox-vis, cmu-vis, wilddash , ox, cmu, cityscapes overwriter img_path, img_ext and save_folder_name. Set to empty string to ignore


    #    'img_path': '%s/%s'%(DATA_ROOT_DIR, CMU_DIR),
    #    'img_path': 'img/',
    #    'img_ext': EXT,
    #    'save_folder_name': 'res',

    #    # specify this if using specific weight file
    #    'network_file': 'pth/from-paper/CMU-CS-Vistas-CE.pth',

    #    'n_slices_per_pass': 10,
    #    'sliding_transform_step': 2 / 3.
    #}

    slice_id = 24
    cam_id = 0
    survey_id = 0
    segment(slice_id, cam_id, survey_id)



