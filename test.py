#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:10:29 2019

@author: YuxuanLong
"""


import os
# import argparse
import random
import numpy as np
import time


import torch
import torch.nn as nn
import cv2
from skimage import io, transform
import torch.utils.data as utils_data

import utils


RUN_ON_GPU = torch.cuda.is_available()
#####test####


def test_single_image(net, file, size = 384, resize = True):
    ##create single image tensor for test in each epoch
    uint_image = io.imread(file)
    test_image_origin = np.array(uint_image).astype(np.float32) / 255.0
    if resize:
        test_image_origin = transform.resize(test_image_origin, (size, size), mode = 'constant', anti_aliasing = True)
        test_image = np.moveaxis(test_image_origin, 2, 0).astype(np.float32) # tensor format  
    else:
        test_image = test_image_origin
        test_image = np.moveaxis(test_image, 2, 0).astype(np.float32) # tensor format  
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = utils.np_to_var(torch.from_numpy(test_image))  
    
    ###dummy test
    pred_test = net.forward(test_image)
    pred_np = utils.var_to_np(pred_test)[0][0]
    
        
    new_mask = (pred_np >= 0.5)

    channel = test_image_origin[:, :, 0]
    channel[new_mask] = 1.0
    test_image_origin[:, :, 0] = channel
    mask = new_mask * 255
    return mask.astype(np.uint8), test_image_origin

def test_single_with_TTA(net, file, size = 384, resize = True):
    ##create single image tensor for test in each epoch
    uint_image = io.imread(file)
    test_image_origin = np.array(uint_image).astype(np.float32) / 255.0

        
    if resize:
        test_image = transform.resize(test_image_origin, (size, size), mode = 'constant', anti_aliasing = True)
    else:
        test_image = test_image_origin


    image_set = []
    for i in range(8):
        b1 = i // 4
        b2 = (i - b1 * 4) // 2
        b3 = i - b1 * 4 - b2 * 2
        tem_image = utils.flip_rotate(test_image, b1, b2, b3, inverse = False)
        tem_image = np.moveaxis(tem_image, 2, 0).astype(np.float32) # tensor format 
        image_set.append(tem_image)
    image_tensor = np.array(image_set)


    image_tensor = utils.np_to_var(torch.from_numpy(image_tensor))  
    
    ###dummy test
    pred_test = net.forward(image_tensor)
    pred_np = utils.var_to_np(pred_test)
    
    pred = np.squeeze(pred_np, axis = 1)
    for i in range(8):
        b1 = i // 4
        b2 = (i - b1 * 4) // 2
        b3 = i - b1 * 4 - b2 * 2
        pred[i] = utils.flip_rotate(pred[i], b1, b2, b3, inverse = True)
        
    pred = np.median(pred, axis = 0)

    new_mask = (pred >= 0.5)

    channel = test_image_origin[:, :, 0]
    channel[new_mask] = 1.0
    test_image_origin[:, :, 0] = channel
    mask = new_mask * 255
    return mask.astype(np.uint8), test_image_origin
    
def test_batch_with_labels(net, file, resize = False, batch_size = 10, image_size = 384, smooth = 1.0, lam = 1.0):
    # On our validation test dataset
    data_augment = False
    rotate = False
    change_color = False
    test_dataset = utils.MyDataset(file, resize, data_augment, image_size, rotate, change_color)
    dataloader = utils_data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)
    epoch_loss = 0.0
    numer = 0.0
    denom = 0.0
    gamma = 0.0
    loss_type = 'bce'
    Loss = utils.loss(smooth, lam, gamma, loss_type)
    for i, batch in enumerate(dataloader):
        print('Test on batch %d'%i)
        image = utils.np_to_var(batch['image'])
        mask = utils.np_to_var(batch['mask'])
        pred = net.forward(image)
        
        loss = Loss.final_loss(pred, mask)
        epoch_loss += loss.data.item() * batch_size
        
        mask = utils.var_to_np(mask)
        pred = utils.var_to_np(pred)
        numer += (mask * (pred > 0.5)).sum()
        denom += mask.sum() + (pred > 0.5).sum()
        
    epoch_loss /= len(test_dataset)
    f1 = 2.0 * numer / denom
    return epoch_loss, f1
