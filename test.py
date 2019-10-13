#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:10:29 2019

@author: YuxuanLong
"""


#test_num = 0
#pred_np = var_to_np(pred)[test_num][0]
#
#new_mask = pred_np >= 0.5
#
#image_up = var_to_np(image)[0]
#
#image_new = np.moveaxis(image_up, 0, 2)
#
#channel = image_new[:, :, 0]
#channel[new_mask] = 1.0
#image_new[:, :, 0] = channel
#
#io.imshow(image_new)



import os
import shutil
# import argparse
import random
import numpy as np
import time

from skimage import io, transform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils

import torch.utils.data as utils_data

import utils

if __name__ == '__main__':
    test_dir = './data/main_data/test'
    
    net = utils.create_models()
    net.load_state_dict(torch.load('./parameters/weights'))
    net.eval()
    
    resize = False
    image_size = 384
        ##create single image tensor for test in each epoch
    test_image_origin = io.imread('./data/main_data/test/test_01.png')
    test_image_origin = utils.image_transform(test_image_origin, resize, image_size)
    test_image_dum = np.moveaxis(test_image_origin, 0, 2)
    test_image = np.expand_dims(test_image_origin, axis = 0)
    test_image = utils.np_to_var(torch.from_numpy(test_image))  
    
    
    
        ###dummy test
    pred_test = net.forward(test_image)
    pred_np = utils.var_to_np(pred_test)[0][0]
    
    new_mask = pred_np >= 0.5
    
    dummy = test_image_dum + 0
    
    channel = dummy[:, :, 0]
    channel[new_mask] = 1.0
    dummy[:, :, 0] = channel
        
    dummy = (dummy * 255).astype(np.uint8)
    io.imsave('./output/test1' + '.png', dummy)