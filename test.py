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

from skimage import io, transform

import torch
import torch.nn as nn


import torch.utils.data as utils_data

import utils

RUN_ON_GPU = torch.cuda.is_available()
#####test####
# return loss over test data, if labels provided
# return output masks over test data
# test for single image

def test_single_image(net, file, size = 384, resize = True):
    ##create single image tensor for test in each epoch
    test_image_origin = io.imread(file)
    test_image_origin = np.array(test_image_origin).astype(np.float32) / 255.0
    if resize:
        test_image_origin = transform.resize(test_image_origin, (size, size), mode = 'constant', anti_aliasing=True)
    
        test_image = utils.image_resize(test_image_origin, resize, size)
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
    
def test_batch_with_labels(net, file, batch_size = 10, image_size = 384, smooth = 1.0, lam = 1.0):
    # On our validation test dataset
    resize = True
    data_augment = False
    rotate = False
    test_dataset = utils.MyDataset(file, resize, data_augment, image_size, rotate)
    dataloader = utils_data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)
    epoch_loss = 0.0
    numer = 0.0
    denom = 0.0
    Loss = utils.loss(smooth, lam)
    for i, batch in enumerate(dataloader):
        print('Test on batch %d'%i)
        image = utils.np_to_var(batch['image'])
        mask = utils.np_to_var(batch['mask'])
        pred = net.forward(image)
        
        loss = Loss.final_loss(pred, mask)
        epoch_loss += loss.data.item() * batch_size
        
        numer += utils.var_to_np(mask * (pred > 0.5)).sum()
        denom += utils.var_to_np(mask).sum() + utils.var_to_np(pred > 0.5).sum()
        
    epoch_loss /= len(test_dataset)
    f1 = 2.0 * numer / denom
    return epoch_loss, f1

def test_batch_without_labels(net, file, batch_size = 5):
    # On the real test dataset
    test_dataset = utils.TestDataset(file)
    dataloader = utils_data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)
    count = 1
    for batch in dataloader:
        image = utils.np_to_var(batch['image'])
        pred = net.forward(image)   
        
        pred = utils.var_to_np(pred)
        for i in range(pred.shape[0]):
            mask = (pred[i][0] > 0.5) * 255
            mask = mask.astype(np.uint8)
            io.imsave('./output/' + 'test' + str(count) + '.png', mask)
            count += 1

if __name__ == '__main__':
    test_dir = './data/main_data/test'
    test_set_output = False
    test_with_labels = False

    net = utils.create_models()
#    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    if RUN_ON_GPU:
        net.load_state_dict(torch.load('./parameters/weights'))
    else:
        net.load_state_dict(torch.load('./parameters/weights', map_location = lambda storage, loc: storage))
    net.eval()
    
    resize = False
    image_size = 384

    if test_set_output:    
        file = './data/main_data/test_set_images/'
        for i in range(1, 51):
            t = 'test_' + str(i)
            name = file + t + '/' + t + '.png'
            mask, image = test_single_image(net, name, size = 384, resize = False)
            io.imsave('./output/' + 'test' + str(i) + '.png', mask)
            
    if test_with_labels:
        file = './data/main_data/training'
        loss, f1 = test_batch_with_labels(net, file, batch_size = 1, image_size = 384, smooth = 1.0, lam = 1.0)
        