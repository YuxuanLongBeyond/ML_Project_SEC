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

def normalize_both(image, mask):
    image = np.array(image).astype(np.float32) / 255.0
    mask = np.array(mask).astype(np.float32) / 255.0
    return image, mask
    
def image_resize(image, resize, size):
    if resize:
        image = transform.resize(image, (size, size), mode = 'constant', anti_aliasing=True)
    image = np.moveaxis(image, 2, 0).astype(np.float32) # tensor format
    return image

def mask_resize(mask, resize, size):
    
    if resize:
        mask = transform.resize(mask, (size, size), mode = 'constant', anti_aliasing=True)
    mask = np.array(mask >= 0.5).astype(np.float32) # test here
    mask = np.expand_dims(mask, axis = 0)
    return mask

def rotate_both(image, mask):
    if np.random.random() < 0.5:
        image = transform.rotate(image, 90)
        mask = transform.rotate(mask, 90)
    return image, mask

def flip_both(image, mask):
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if np.random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask
       
class TestDataset(utils_data.Dataset):
    def __init__(self, root):
        self.root = root
        
        self.file_list = []
        for f in os.listdir(root):
            name = os.path.join(root, f)
            if not os.path.isfile(name):
                
                self.file_list.append(os.path.join(name, f) + '.png')
        self.file_list.sort()
#        self.file_list = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
#        random.shuffle(self.mask_file_list)
        

    def __getitem__(self, index):
        img_name = os.path.join(self.file_list[index])
        
        image = io.imread(img_name)
        image = np.array(image).astype(np.float32) / 255.0
        image = image_resize(image, False, 0)
        
        sample = {'image': image}

        return sample
  
    def __len__(self):
        return len(self.file_list) 

class MyDataset(utils_data.Dataset):
    def __init__(self, root, resize, data_augment, size):
        self.size = size
        self.root = root
        self.data_augment = data_augment
        mask_dir = root + '/groundtruth'
        self.resize = resize
#        self.mask_file_list = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
        self.mask_file_list = [f for f in os.listdir(mask_dir) if 'sat' in f and 'png' in f]
#        random.shuffle(self.mask_file_list)

    def __getitem__(self, index):
        file_name =  self.mask_file_list[index].rsplit('.', 1)[0]
        img_name = self.root + '/images/' + file_name+'.png'
        mask_name = self.root + '/groundtruth/' + file_name+'.png'
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        image, mask = normalize_both(image, mask)
        
        if self.data_augment:
            image, mask = rotate_both(image, mask)
            image, mask = flip_both(image, mask)
        
        # resize and convert to tensor
        image = image_resize(image, self.resize, self.size)
        mask = mask_resize(mask, self.resize, self.size)
        
        sample = {'image': image, 'mask': mask}

        return sample
  
    def __len__(self):
        return len(self.mask_file_list)
    
class MyNewDataset(utils_data.Dataset):
    def __init__(self, root, resize, data_augment, size):
        self.size = size
        self.root = root
        self.resize = resize
        self.data_augment = data_augment
        image_file_list = []
        label_file_list = []
        for f in os.listdir(root):
            if os.path.isfile(os.path.join(root, f)):
                if 'image' in f:
                    image_file_list.append(f)
                if 'labels' in f:
                    label_file_list.append(f)
        image_file_list.sort()
        label_file_list.sort()
        file_num = len(image_file_list)
        self.file_num = file_num
        file_list = [[image_file_list[i], label_file_list[i]] for i in range(file_num)]
        self.file_list = file_list
#        random.shuffle(self.file_list)

    def __getitem__(self, index):
        img_name = self.root + '/' + self.file_list[index][0]
        mask_name = self.root + '/' + self.file_list[index][1]
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        mask = (mask[:, :, 0] == 0).astype(np.float32)
        image = np.array(image).astype(np.float32) / 255.0
        
        if self.data_augment:
#            image, mask = crop_both(image, mask)
            image, mask = rotate_both(image, mask)
            image, mask = flip_both(image, mask)
        
        # resize and convert to tensor
        image = image_resize(image, self.resize, self.size)
        mask = mask_resize(mask, self.resize, self.size)
        
        sample = {'image': image, 'mask': mask}

        return sample
  
    def __len__(self):
        return self.file_num


def get_data_loader(root, new_data = True, resize = True, data_augment = True,
                    image_size = 384, batch_size=100):
    """Creates training data loader."""
    if new_data:
        train_dataset = MyNewDataset(root, resize, data_augment, image_size)
    else:
        train_dataset = MyDataset(root, resize, data_augment, image_size)
    return utils_data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)


def create_models():

    net = model.LinkNet()

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net


def loss(smooth, lam):
    return model.Loss(smooth, lam)
