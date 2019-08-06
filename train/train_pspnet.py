
import datetime
import os, sys, argparse
import numpy as np
from math import sqrt

import torch
import torchvision.transforms as standard_transforms
from torch import optim

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from layers.feature_loss import FeatureLoss
from utils.validator import Validator
from layers.corr_class_loss import CorrClassLoss
from utils.misc import (check_mkdir, AverageMeter, freeze_bn, get_global_opts,
        rename_keys_to_match, get_latest_network_name,
        clean_log_before_continuing)
from models import model_configs
from datasets import cityscapes, correspondences, lake
import utils.corr_transforms as corr_transforms
import utils.transforms as extended_transforms
import utils.joint_transforms as joint_transforms
import datasets.dataset_configs as data_configs
from models import pspnet

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

n_classes = 19

def train_with_correspondences(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    res_dir = 'res/%d'%args.trial
    log_dir = 'res/%d/log'%args.trial
    snap_dir = 'res/%d/snap'%args.trial
    val_dir = 'res/%d/val'%args.trial

    writer = SummaryWriter(log_dir)

    # Network and weight loading
    net = pspnet.PSPNet().to(device)
    if args.snapshot == '':  # If start from beginning
        state_dict = torch.load(args.startnet)
        # needed since we slightly changed the structure of the network in pspnet
        state_dict = rename_keys_to_match(state_dict)
        net.load_state_dict(state_dict)  # load original weights
        print('OK: Load net from %s'%args.startnet)
        start_iter = 0
        best_record = { 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    net.train()
    freeze_bn(net) # TODO: check relevance

    
    # data transforms
    crop_size = 713
    model_config = model_configs.PspnetCityscapesConfig()
    sliding_crop_im = joint_transforms.SlidingCropImageOnly(
        crop_size, args.stride_rate)
    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform
    target_transform = extended_transforms.MaskToTensor()
    train_joint_transform_seg = joint_transforms.Compose([
        joint_transforms.Resize(1024),
        joint_transforms.RandomRotate(10),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomCrop(crop_size)
    ])

    
    # load datasets
    seg_set = lake.Lake('train', args.data_id, args.img_root_dir, args.seg_root_dir,
        joint_transform=train_joint_transform_seg,
        sliding_crop=None,
        transform=input_transform,
        target_transform=target_transform)
    seg_loader = DataLoader( seg_set, batch_size=args.train_batch_size,
            num_workers=args.n_workers, shuffle=True)

    val_set = lake.Lake('val', args.data_id, args.img_root_dir, args.seg_root_dir, 
        joint_transform=train_joint_transform_seg,
        sliding_crop=sliding_crop_im,
        transform=input_transform,
        target_transform=target_transform,
        transform_before_sliding=pre_validation_transform)
    val_loader = DataLoader( val_set, batch_size=1,
            num_workers=args.n_workers, shuffle=False)
    validator = Validator( val_loader, n_classes=n_classes,
            save_snapshot=False, extra_name_str='lake')
    

    # loass
    seg_loss_fct = torch.nn.CrossEntropyLoss( reduction='elementwise_mean',
            ignore_index=lake.ignore_label).to(device)

    # Optimizer setup
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum, nesterov=True)


    # save args
    args_f = open('%s/args.txt'%log_dir, 'w')
    args_f.write(str(args) + '\n\n')
    args_f.close()

    if args.snapshot == '':
        f_handle = open('%s/log.log'%log_dir, 'w', buffering=1)
    else:
        clean_log_before_continuing( '%s/log.log'%log_dir, start_iter)
        f_handle = open('%s/log.log'%log_dir, 'a', buffering=1)

    ##########################################################################
    #
    #       MAIN TRAINING CONSISTS OF ALL SEGMENTATION LOSSES AND A CORRESPONDENCE LOSS
    #
    ##########################################################################
    softm = torch.nn.Softmax2d()

    val_iter = 0
    seg_loss_meters = AverageMeter()
    curr_iter = start_iter

    for i in range(args.max_iter):
        optimizer.param_groups[0]['lr'] = 2 * args.lr * (1 - float(curr_iter) / args.max_iter) ** args.lr_decay
        optimizer.param_groups[1]['lr'] = args.lr * (1 - float(curr_iter) / args.max_iter) ** args.lr_decay

        #######################################################################
        #       SEGMENTATION UPDATE STEP
        #######################################################################
        #
        # get segmentation training sample
        inputs, gts = next(iter(seg_loader))
        slice_batch_pixel_size = inputs.size( 0) * inputs.size(2) * inputs.size(3)
        inputs = inputs.to(device)
        gts = gts.to(device)

        optimizer.zero_grad()
        outputs, aux = net(inputs)
        main_loss = args.seg_loss_weight * seg_loss_fct(outputs, gts)
        aux_loss = args.seg_loss_weight * seg_loss_fct(aux, gts)
        loss = main_loss + 0.4 * aux_loss
        loss.backward()
        optimizer.step()
        
        seg_loss_meters.update( main_loss.item(), slice_batch_pixel_size)

        #######################################################################
        #       LOGGING ETC
        #######################################################################
        curr_iter += 1
        val_iter += 1

        writer.add_scalar('train_seg_loss_cs', train_seg_cs_loss.avg,  curr_iter)
        writer.add_scalar('train_seg_loss_extra', train_seg_extra_loss.avg, curr_iter)
        writer.add_scalar('train_seg_loss_vis', train_seg_vis_loss.avg, curr_iter)
        writer.add_scalar('train_corr_loss', train_corr_loss.avg, curr_iter)
        writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

        if (i + 1) % args.print_freq == 0:
            str2write = '[iter %d / %d], [seg_loss %.5f], [lr %.10f]' % (
                curr_iter, len(seg_loader), seg_loss_meters.avg, optimizer.param_groups[1]['lr'])
            print(str2write)
            f_handle.write(str2write + "\n")

        if val_iter % args.val_interval==0:
            validator.run( net, optimizer, args, curr_iter, val_dir,
                     f_handle, writer=writer)

    # Post training
    f_handle.close()
    writer.close()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--startnet', type=str)
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--seg_loss_weight', type=float)
    parser.add_argument('--print_freq', type=int)
    parser.add_argument('--val_interval', type=int)
    parser.add_argument('--stride_rate', type=float)
    parser.add_argument('--n_workers', type=int)

    parser.add_argument('--img_root_dir', type=str)
    parser.add_argument('--seg_root_dir', type=str)
    parser.add_argument('--data_id', type=int)

    args = parser.parse_args()

    train_with_correspondences(args)
