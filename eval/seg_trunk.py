

from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes
from utils.misc import rename_keys_to_match

import os, sys, time, re, h5py, math
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as standard_transforms


# colmap output dir
MACHINE = 1
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


def run_net(filenames_ims, filenames_segs):

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
        #print(save_path)
        segmentor.run_and_save(
            im_file,
            save_path,
            '',
            pre_sliding_crop_transform = pre_validation_transform,
            sliding_crop = sliding_crop,
            input_transform = input_transform,
            skip_if_seg_exists = True,
            use_gpu = True,
            save_logits=False)
        count += 1
        #if count == 3:
        #    break


def segment():

    fn_l = sorted(os.listdir('meta/trunks/img/'))
    filenames_ims = ['meta/trunks/img/%s'%l for l in fn_l]
    
    save_folder = 'meta/trunks/seg/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filenames_segs = ['%s/%s.png'%(save_folder, l) for l in fn_l]
    run_net(filenames_ims, filenames_segs)


if __name__ == '__main__':

    segment()


