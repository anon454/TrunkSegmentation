import datetime
import os, sys, argparse
import numpy as np
from math import sqrt
import cv2
import time
import PIL.Image
import pickle

from models import model_configs
import torch
import torchvision.transforms as standard_transforms
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.segmentor import Segmentor

from utils.validator import Validator
from utils.misc import AverageMeter, freeze_bn, rename_keys_to_match
from utils.misc import evaluate_incremental
from datasets import lake, cityscapes
from models import pspnet
import tools
import datasets.test_data as test_data
import utils.joint_transforms as joint_transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

n_classes = 19
ignore_label = 255

def run_net(filenames_img, filenames_segs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = [args.train_crop_size, args.train_crop_size]
    net = pspnet.PSPNet(input_size = input_size).to(device)
    if not os.path.exists(args.trained_net):
        print("Error: no trained net to evaluate")
        exit(1)
    state_dict = torch.load(args.trained_net)
    #needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    net.load_state_dict(state_dict)
    print('OK: Load net from %s'%args.trained_net)

    start_iter = 0
    net.eval()

    #data proc
    model_config = model_configs.PspnetCityscapesConfig()
    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform

    sliding_crop = joint_transforms.SlidingCropImageOnly(args.val_crop_size,
            2/3.)
    
    # encapsulate pytorch model in Segmentor class
    print("Class number: %d"%net.n_classes)
    segmentor = Segmentor(net, net.n_classes, colorize_fcn =
            cityscapes.colorize_mask, n_slices_per_pass = 10)

    #let's go
    count = 1
    t0 = time.time()
    for i, im_file in enumerate(filenames_img):
        save_path = filenames_segs[i]
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_img),
                        tnow - t0, (tnow - t0) / count * len(filenames_img),
                        im_file))

        print(save_path)

        segmentor.run_and_save(im_file, save_path, '',
                pre_sliding_crop_transform = pre_validation_transform,
                sliding_crop = sliding_crop, input_transform = input_transform,
                skip_if_seg_exists = True, use_gpu = True, save_logits=False)
        count += 1

def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    res_dir = 'res/%d'%args.trial
    log_dir = 'res/%d/log'%args.trial
    snap_dir = 'res/%d/snap'%args.trial
    val_dir = 'res/%d/val'%args.trial

    img_fn_v = np.loadtxt('meta/list/data/%d/all_img.txt'%args.data_id, dtype=str)
    filenames_img = ['%s/%s'%(args.img_root_dir, l) for l in img_fn_v]
    filenames_segs = ['%s/%s'%(args.seg_save_root, l) for l in img_fn_v]

    run_net(filenames_img, filenames_segs)

if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--trial', type=int)

        # train
        # log
        parser.add_argument('--trained_net', type=str)


        # data
        parser.add_argument('--data_id', type=int)
        parser.add_argument('--img_root_dir', type=str)
        #parser.add_argument('--seg_root_dir', type=str)
        parser.add_argument('--val_crop_size', type=int)
        parser.add_argument('--train_crop_size', type=int)
        parser.add_argument('--stride_rate', type=float)
        parser.add_argument('--n_workers', type=int)
        parser.add_argument('--seg_save_root', type=str)

                                  
        args = parser.parse_args()

        test(args)


