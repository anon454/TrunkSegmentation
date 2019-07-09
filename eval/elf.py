from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes, dataset_configs
from utils.misc import check_mkdir, get_global_opts, rename_keys_to_match

import os
import re
import numpy as np
import torch
from torch.autograd import Variable, grad
from PIL import Image
import torchvision.transforms as standard_transforms
import h5py
import math

import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def _pad(img, crop_size=713):
    h, w = img.shape[: 2]
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    return img, h, w

def segment_images_in_folder_for_experiments(network_folder, args):
    # Predefined image sets and paths
    dset = dataset_configs.CmuConfig()
    args['img_path'] = dset.test_im_folder
    args['img_ext'] = dset.im_file_ending
    img_folder = dset.test_im_folder
    
    save_folder = 'res/'
    check_mkdir(save_folder)
    
    # /home/abenbihi/ws/datasets/correspondence/CMU/segmented_images/testing/imgs
    print('img_path: %s'%args['img_path'])
    
    # network model
    print("Loading specified network")
    slash_inds = [i for i in range(
        len(args['network_file'])) if args['network_file'].startswith('/', i)]
    network_folder = args['network_file'][:slash_inds[-1]]
    network_file = args['network_file']

    print('network_folder: %s\nnetwork_file: %s'%(network_folder,
        network_file))
    #exit(0)

    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network().to(device)
    
    print('load model ' + network_file)
    state_dict = torch.load(
        network_file,
        map_location=lambda storage,
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
        713, args['sliding_transform_step'])

    # get all file names
    filenames_ims = list() # input files
    filenames_segs = list() # output files
    print('Scanning %s for images to segment.' % img_folder)
    for root, subdirs, files in os.walk(img_folder):
        filenames = [f for f in files if f.endswith(args['img_ext'])]
        if len(filenames) > 0:
            print('Found %d images in %s' % (len(filenames), root))
            filenames_ims += [os.path.join(root, f) for f in filenames]
            filenames_segs += [os.path.join(save_folder, # output files
                                            f.replace(args['img_ext'],
                                                      '.png')) for f in filenames]
    
    # encapsulate pytorch model in Segmentor class
    print("Class number: %d"%net.n_classes) # 19
    #segmentor = Segmentor(
    #        net,
    #        net.n_classes,
    #        colorize_fcn=cityscapes.colorize_mask,
    #        n_slices_per_pass=args['n_slices_per_pass'])

    # let's go
    count = 1
    t0 = time.time()
    for im_file, save_path in zip(filenames_ims, filenames_segs):
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))

        try:
            img = Image.open(im_file).convert('RGB')
        except OSError:
            print("Error reading input image, skipping: {}".format(im_file))

        # creating sliding crop windows and transform them
        img_size_orig = img.size
       
        #if pre_sliding_crop_transform is not None:  # might reshape image
        #    img = pre_sliding_crop_transform(img)

        w, h = img.size
        long_size = max(h, w)
        img = np.array(img)
        img, sub_h, sub_w = _pad(img)
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')

        #img_slices, slices_info = sliding_crop(img)
        img_slices = [input_transform(img)]
        img_slices = torch.stack(img_slices, 0)
        img_slices = img_slices.to(device)
        output = net(img_slices)

        toto = grad(output, img_slices)


        #slices_info = torch.LongTensor(slices_info)
        #slices_info.squeeze_(0)


        #segmentor.run_and_save(
        #    im_file,
        #    save_path,
        #    pre_sliding_crop_transform=pre_validation_transform,
        #    sliding_crop=sliding_crop,
        #    input_transform=input_transform,
        #    skip_if_seg_exists=True,
        #    use_gpu=args['use_gpu'])
        count += 1
        if count == 3:
            break


    exit(0)


if __name__ == '__main__':
    global_opts = get_global_opts()

    DATA_ROOT_DIR = '/home/abenbihi/ws/datasets/correspondence/'
    
    CMU_DIR = 'CMU/segmented_images/testing/imgs/' 
    EXT = '.png'

    network_folder = '' # not used for now because I specify the network file

    args = {
        'use_gpu': True,
        # 'miou' (miou over classes present in validation set), 'acc'
        'validation_metric': 'miou',
        'img_set': 'cmu',  # ox-vis, cmu-vis, wilddash , ox, cmu, cityscapes overwriter img_path, img_ext and save_folder_name. Set to empty string to ignore


        'img_path': '%s/%s'%(DATA_ROOT_DIR, CMU_DIR),
        'img_path': 'img/',
        'img_ext': EXT,
        'save_folder_name': 'res',

        # specify this if using specific weight file
        'network_file': 'pth/from-paper/CMU-CS-Vistas-CE.pth',

        'n_slices_per_pass': 10,
        'sliding_transform_step': 2 / 3.
    }

    segment_images_in_folder_for_experiments(network_folder, args)
