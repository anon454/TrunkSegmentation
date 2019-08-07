
import datetime
import os, sys, argparse
import numpy as np
from math import sqrt

import torch
import torchvision.transforms as standard_transforms
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.validator import Validator
from utils.misc import AverageMeter, freeze_bn, rename_keys_to_match
from datasets import lake
from models import pspnet

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

n_classes = 19

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    res_dir = 'res/%d'%args.trial
    log_dir = 'res/%d/log'%args.trial
    snap_dir = 'res/%d/snap'%args.trial
    val_dir = 'res/%d/val'%args.trial


    # Network and weight loading
    input_size = [args.train_crop_size, args.train_crop_size]
    net = pspnet.PSPNet(input_size=input_size).to(device)
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

    # loss
    seg_loss_fct = torch.nn.CrossEntropyLoss( reduction='elementwise_mean',
            ignore_index=lake.ignore_label).to(device)
    softm = torch.nn.Softmax2d()

    # Optimizer setup
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum, nesterov=True)
  

    # load datasets
    seg_set = lake.Lake(args, 'train')
    seg_loader = DataLoader( seg_set, batch_size=args.batch_size,
            num_workers=args.n_workers, shuffle=True)

    val_set = lake.Lake(args, 'val')
    val_loader = DataLoader( val_set, batch_size=1,
            num_workers=args.n_workers, shuffle=True)
    validator = Validator( val_loader, n_classes=n_classes,
            save_snapshot=False, extra_name_str='lake')
    

    # Set log and summary
    writer = SummaryWriter(log_dir)
    
    args_f = open('%s/args.txt'%log_dir, 'w')
    args_f.write(str(args) + '\n\n')
    args_f.close()

    if args.snapshot == '':
        f_handle = open('%s/log.log'%log_dir, 'w', buffering=1)
    else:
        clean_log_before_continuing( '%s/log.log'%log_dir, start_iter)
        f_handle = open('%s/log.log'%log_dir, 'a', buffering=1)


    
    # let's go
    val_iter = 0
    seg_loss_meters = AverageMeter()
    curr_iter = start_iter

    max_iter = int(args.max_epoch * len(seg_loader) / args.batch_size)

    for epoch in range(args.max_epoch):
        for batch_idx, batch in enumerate(seg_loader):
            optimizer.param_groups[0]['lr'] = 2 * args.lr * (1 - float(curr_iter) / max_iter) ** args.lr_decay
            optimizer.param_groups[1]['lr'] = args.lr * (1 - float(curr_iter) / max_iter) ** args.lr_decay

            # get segmentation training sample
            inputs, gts = batch # next(iter(seg_loader))
            inputs, gts = inputs.to(device), gts.to(device)
            #slice_batch_pixel_size = inputs.size( 0) * inputs.size(2) * inputs.size(3)

            optimizer.zero_grad()
            outputs, aux = net(inputs)

            print('inputs.shape', inputs.size())
            print('gts.shape', gts.size())
            print('outputs.shape', outputs.size())

            break

            main_loss =  seg_loss_fct(outputs, gts)
            aux_loss = seg_loss_fct(aux, gts)
            loss = main_loss + 0.4 * aux_loss
            loss.backward()
            optimizer.step()
            
            curr_iter += 1
            val_iter += 1

            
            seg_loss_meters.update( main_loss.item(), slice_batch_pixel_size)
            if curr_iter % args.summary_interval:
                writer.add_scalar('train_seg_loss_cs', seg_loss_meters.avg,  curr_iter)
                writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)
            if curr_iter % args.log_interval == 0:
                str2write = 'Epoch %d/%d\tIter %d\tLoss: %.5f\tlr %.10f' % (
                    epoch, args.max_epoch, curr_iter, len(seg_loader), seg_loss_meters.avg, optimizer.param_groups[1]['lr'])
                print(str2write)
                f_handle.write(str2write + "\n")
            
        break

        # validation
        if epoch % args.val_interval==0:
            eval_iter_max = 10
            validator.run( net, optimizer, best_record, curr_iter, res_dir,
                     f_handle, eval_iter_max, writer=writer)

    # Post training
    f_handle.close()
    writer.close()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int)

    # train
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--momentum', type=float)
    
    # log
    parser.add_argument('--startnet', type=str)
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--log_interval', type=int)
    parser.add_argument('--summary_interval', type=int)
    parser.add_argument('--val_interval', type=int)

    
    # data
    parser.add_argument('--data_id', type=int)
    parser.add_argument('--img_root_dir', type=str)
    parser.add_argument('--seg_root_dir', type=str)
    parser.add_argument('--val_crop_size', type=int)
    parser.add_argument('--train_crop_size', type=int)
    parser.add_argument('--stride_rate', type=float)
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--random_rotate', type=int)
    parser.add_argument('--rot_max', type=int, help='in degrees')
    parser.add_argument('--random_crop', type=int)
    parser.add_argument('--random_flip', type=int)
    parser.add_argument('--data_debug', type=int)


    args = parser.parse_args()

    train(args)
