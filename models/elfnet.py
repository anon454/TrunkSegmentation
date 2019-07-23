#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15
#
# Modified by: Måns Larsson, 2019


from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import _ConvBatchNormReLU, _ResBlock
from models.pspnet import PSPNet 
#import models.pspnet as pspnet

class ELFNet(nn.Module):
    """Pyramid Scene Parsing Network"""
    # DEFAULTS ARE FOR CITYSCAPES

    def __init__(self, fcn):
        super(ELFNet, self).__init__()
        
        # pre resnet
        print(fcn._modules['layer1']._modules) # I hate pytorch
        print(fcn._modules['layer1']._modules['conv1']._modules)

        #self.conv1 =fcn._modules['layer1']._modules['conv1']._modules['conv']

        # layer1
        self.l1_conv1 = fcn._modules['layer1']._modules['conv1'] # I hate pytorch
        self.l1_conv2 = fcn._modules['layer1']._modules['conv2'] # I hate pytorch
        self.l1_conv3 = fcn._modules['layer1']._modules['conv3'] # I hate pytorch
        self.l1_pool1 = fcn._modules['layer1']._modules['pool'] # I hate pytorch
        #MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        print(self.l1_pool1)
        
        # layer2
        self.l2_block1 = fcn._modules['layer2']._modules['block1']
        self.l2_block2 = fcn._modules['layer2']._modules['block2']
        self.l2_block3 = fcn._modules['layer2']._modules['block3']

        
        self.gradients = None
        def hook_function(module, grad_in, grad_out):
            print('grad_in.shape', grad_in[0].size()) # feature map
            print('grad_out.shape', grad_out[0].size()) # gradient
            self.gradients = grad_out[0]
            # register hook to last feature map
        #feat_list = self.fcn._modules
        #input(feat_list)
        self.l2_block3.register_backward_hook(hook_function)

    def forward(self, x):
        
        l1_conv1 = self.l1_conv1(x)
        l1_conv2 = self.l1_conv2(l1_conv1)
        l1_conv3 = self.l1_conv3(l1_conv2)
        l1_pool1 = self.l1_pool1(l1_conv3)

        l2_block1 = self.l2_block1(l1_pool1)
        l2_block2 = self.l2_block2(l2_block1)
        l2_block3 = self.l2_block3(l2_block2)

        return l2_block3



if __name__ == '__main__':
    model = PSPNet(n_classes=19, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1])
    print(list(model.named_children()))
    
    elfnet = ELFNet(model.fcn)
    #model.eval()
    elfnet.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 713, 713))
    print(model(image).size())
    print(elfnet(image).size())



