from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes, dataset_configs
from utils.misc import check_mkdir, get_global_opts, rename_keys_to_match
from models import pspnet, elfnet

import os, re
import sys, time
from PIL import Image
import numpy as np
import h5py, math
import cv2

import torch
from torch.autograd import Variable, grad
import torchvision.transforms as standard_transforms


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def _pad(img, crop_size=713):
    h, w = img.shape[: 2]
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    return img, h, w


def segment_images_in_folder_for_experiments(network_folder, args):

    DATA_ROOT_DIR = '/home/gpu_user/assia/ws/datasets/'
    CMU_DATA_DIR = '%s/correspondence/CMU/'%DATA_ROOT_DIR 
    img_dir = '%s/segmented_images/testing/imgs/'%CMU_DATA_DIR
    res_dir = 'res/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    # /home/abenbihi/ws/datasets/correspondence/CMU/segmented_images/testing/imgs
    print('img_dir: %s'%img_dir)


    # network model
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # Network and weight loading
    pspNet = pspnet.PSPNet().to(device) 
    print("Loading specified network")
    #print('network_folder: %s\nnetwork_file: %s'%(network_folder, network_file))
    slash_inds = [i for i in range(
        len(args['network_file'])) if args['network_file'].startswith('/', i)]
    network_folder = args['network_file'][:slash_inds[-1]] # pth/from-paper/
    network_file = args['network_file'] # pth/from-paper/CMU-CS-Vistas-CE.pth
    print('load model ' + network_file)
    state_dict = torch.load( network_file, map_location=lambda storage, loc: storage)
    # needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    pspNet.load_state_dict(state_dict)
    pspNet.eval()


    # elf net
    elfNet = elfnet.ELFNet(pspNet.fcn)
    elfNet.eval()

    
    t0 = time.time()
    count = 1
    img_fn_l = sorted(os.listdir(img_dir))
    img_num = len(img_fn_l)

    for img_root_fn in img_fn_l:
        img_fn = '%s/%s'%(img_dir, img_root_fn)
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, img_num, 
            tnow - t0, (tnow - t0) / count * img_num, img_root_fn))

        img = cv2.imread(img_fn)
        #img = img[:256, :256]

        imgf = img.astype(np.float32)
        imgf -= np.array([123.68, 116.779, 103.939])
        imgf = np.expand_dims(imgf.transpose((2,0,1)),0) # bz, c, h, w
        img_t = torch.Tensor(imgf)
        #img_v = Variable(img_t, requires_grad=False).cuda()
        img_v = Variable(img_t).cuda().requires_grad_()
        print('img.shape', img_v.size())

        output = elfNet(img_v)
        output_np = output.cpu().data.numpy()
        #print(output_np)
        print('pool1.shape', output_np.shape) # 1,3,128,128
        
        elfNet.zero_grad()
        output.backward(gradient=output, retain_graph=True)
        grad_arr = elfNet.gradients.cpu().data.numpy()[0]
        print('grad_arr.shape', grad_arr.shape) # 128, 128, 128

        print('img_v.is_leaf', img_v.is_leaf)
        print('output.is_leaf', output.is_leaf)
 
        toto = torch.autograd.backward(output, output)
    
        #print(img_v.grad)
        print('img_v.grad.shape: ', img_v.grad.size())
        grad = img_v.grad.cpu().data.numpy().squeeze()
        print(grad.shape)
        grad = np.transpose(grad, (1,2,0))
        grad = np.abs(grad)
        grad = np.mean(grad, axis=2)
        grad /= np.max(grad)
        cv2.imshow('img', img)
        cv2.imshow('grad', grad)
        cv2.waitKey(0)
        #print(toto)
        #toto_np = toto.cpu().data.numpy()
        #print('toto_np.shape', toto_np.shape)
       
        
        ## ko
        #toto = grad(output, img_v)
        #toto_np = toto.cpu().data.numpy()
        #print('toto_np.shape', toto_np.shape)

        count += 1
        if count == 10:
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
