#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:06:57 2019

@author: YuxuanLong
"""

import os
# import argparse
import random
import numpy as np

from skimage import io, transform

import torch
from torch.autograd import Variable
import torchvision.utils

import torch.utils.data as utils_data

import model

# Run on GPU if CUDA is available.
RUN_ON_GPU = torch.cuda.is_available()

def np_to_var(x):
    """Converts numpy to variable."""
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    """Converts variable to numpy."""
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def image_transform(image, resize, size):
    if resize:
        image = transform.resize(image, (size, size), mode = 'constant', anti_aliasing=True)
        image = np.array(image)
    else:
        image = np.array(image) / 255.0
    
    image = np.moveaxis(image, 2, 0) # tensor format
    image = image.astype(np.float32)
    return image
        
def mask_transform(mask, resize, size):
    if resize:
        mask = transform.resize(mask, (size, size), mode = 'constant', anti_aliasing=True)
    else:
        mask = np.array(mask) / 255.0
    mask = np.array(mask >= 0.5).astype(np.float32) # test here
    mask = np.expand_dims(mask, axis = 0)
    return mask

class MyDataset(utils_data.Dataset):
    def __init__(self, root, resize = None, size = 384):
        self.size = size
        self.root = root
        mask_dir = root + '/groundtruth'
        self.resize = resize
        self.mask_file_list = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
        random.shuffle(self.mask_file_list)

    def __getitem__(self, index):
        file_name =  self.mask_file_list[index].rsplit('.', 1)[0]
        img_name = self.root + '/images/' + file_name+'.png'
        mask_name = self.root + '/groundtruth/' + file_name+'.png'
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        image = image_transform(image, self.resize, self.size)
        mask = mask_transform(mask, self.resize, self.size)
        
        sample = {'image': image, 'mask': mask}

        return sample
  
    def __len__(self):
        return len(self.mask_file_list)


def get_data_loader(root, resize = True, image_size = 384, batch_size=10):
    """Creates training data loader."""
    train_dataset = MyDataset(root, resize, image_size)
    return utils_data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)


def create_models():

    net = model.LinkNet()

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net


def loss(smooth, lam, beta):
    return model.Loss(smooth, lam, beta)
