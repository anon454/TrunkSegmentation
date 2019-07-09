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
       
        self.conv1 = fcn._modules['layer1']._modules['conv1'] # I hate pytorch
        self.conv2 = fcn._modules['layer1']._modules['conv2'] # I hate pytorch
        self.conv3 = fcn._modules['layer1']._modules['conv3'] # I hate pytorch
        self.pool1 = fcn._modules['layer1']._modules['pool'] # I hate pytorch
        #MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        print(self.pool1)
        
        self.gradients = None
        def hook_function(module, grad_in, grad_out):
            print('grad_in.shape', grad_in[0].size()) # feature map
            print('grad_out.shape', grad_out[0].size()) # gradient
            self.gradients = grad_out[0]
            # register hook to last feature map
        #feat_list = self.fcn._modules
        #input(feat_list)
        self.pool1.register_backward_hook(hook_function)

    def forward(self, x):
        x_size = x.size()
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        h = self.pool1(conv3)
        return h



if __name__ == '__main__':
    model = PSPNet(n_classes=19, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1])
    print(list(model.named_children()))
    
    elfnet = ELFNet(model.fcn)
    #model.eval()
    elfnet.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 713, 713))
    print(model(image).size())
    print(elfnet(image).size())



