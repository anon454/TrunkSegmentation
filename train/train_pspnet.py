
import datetime
import os, sys, argparse
import numpy as np
from math import sqrt
import cv2

import torch
import torchvision.transforms as standard_transforms
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.validator import Validator
from utils.misc import AverageMeter, freeze_bn, rename_keys_to_match
from datasets import lake
from models import pspnet
import tools
import datasets.test_data as test_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

n_classes = 2

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

    # Optimizer setup
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum, nesterov=True)
  

    # load datasets
    seg_set = lake.Lake(args, 'train')
    seg_loader = DataLoader( seg_set, batch_size=args.batch_size, num_workers =args.n_workers, shuffle=True)

    val_set = lake.Lake(args, 'val')
    val_loader = DataLoader( val_set, batch_size=1,
           num_workers =args.n_workers, shuffle=True)
    validator = Validator( val_loader, n_classes=n_classes,
            save_snapshot=True, extra_name_str='lake')
    

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
        
    mean_std = ([-116.779/255.0, -103.939/255., -123.68/255.], [1, 1, 1])
    normalize_back = standard_transforms.Normalize(*mean_std)

    for epoch in range(args.max_epoch):
        for batch_idx, batch in enumerate(seg_loader):
            optimizer.param_groups[0]['lr'] = 2 * args.lr * (1 - float(curr_iter) / max_iter) ** args.lr_decay
            optimizer.param_groups[1]['lr'] = args.lr * (1 - float(curr_iter) / max_iter) ** args.lr_decay

            # get segmentation training sample
            inputs, gts = batch # next(iter(seg_loader))
            #print("original gt", np.unique(gts))
            inputs, gts = inputs.to(device), gts.to(device)
            #slice_batch_pixel_size = inputs.size( 0) * inputs.size(2) * inputs.size(3)
            optimizer.zero_grad()
            outputs, aux = net(inputs)

           # print('inputs.shape', inputs.size())
           # print('gts.shape', gts.size())
           # print('outputs.shape', outputs.size())

            main_loss = seg_loss_fct(outputs, gts)
            aux_loss = seg_loss_fct(aux, gts)
            loss = main_loss + 0.4 * aux_loss
            loss.backward()
            optimizer.step()
            
            curr_iter += 1
            val_iter += 1
        
            #seg_loss_meters.update( main_loss.item(), slice_batch_pixel_size)
            if curr_iter % args.summary_interval:
                writer.add_scalar('train_loss', loss,  curr_iter)
                writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)
                
                output_np = outputs[0,:,:,:].max(0)[1].squeeze_(0).cpu().numpy()
                output_np_copy = output_np.copy()
                for k, v in seg_set.id_to_trainid.items():
                    output_np_copy[output_np == v] = k
                output_np = output_np_copy
                #print('output_np.shape', output_np.shape)
                palette = [[128, 64,128],
                        [244, 32, 232],
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
                palette_bgr = [ [l[2], l[1], l[0]] for l in palette]
                output_col = test_data.lab2col(output_np, palette_bgr)
                print("output",np.unique(output_np))
                #cv2.imshow('output_col', output_col)
                #cv2.waitKey(0)
                output_col = output_col[:,:,::-1]
                #output_col /= np.max(output_col)
                output_col = np.transpose(output_col, (2,0,1))


                gt_np = np.squeeze(gts.data.cpu().numpy()[0,:,:])
                gt_np_copy = gt_np.copy()
                #print(np.unique(gt_np_copy))
                for k, v in seg_set.id_to_trainid.items():
                    gt_np_copy[gt_np == v] = k
                gt_np = gt_np_copy
                #print("gt",np.unique(gt_np))
                gt_col = test_data.lab2col(gt_np, palette_bgr)
                gt_col = gt_col[:,:,::-1] # bgr -> rgb for tensorboard
                #cv2.imshow('gt_col', gt_col)
                #cv2.waitKey(0)
                #gt_col /= np.max(gt_col)
                gt_col = np.transpose(gt_col, (2,0,1))

                #inputs_np = inputs.data.cpu().numpy()[0,:,:,:]
                #inputs_np = np.transpose(inputs_np, (1,2,0))
                #inputs /= inputs.max()
                #inputs = normalize_back(inputs.data)

                writer.add_image('img', inputs[0,:,:,:])
                writer.add_image('pred', output_col)
                writer.add_image('gt', gt_col)


            if curr_iter % args.log_interval == 0:
                str2write = 'Epoch %d/%d\tIter %d\tLoss: %.5f\tlr %.10f' % (
                    epoch, args.max_epoch, curr_iter, loss.data.cpu().numpy(), optimizer.param_groups[1]['lr'])
                print(str2write)
                f_handle.write(str2write + "\n")
            
        # validation
        if epoch % args.val_interval==0 and epoch !=0:
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
