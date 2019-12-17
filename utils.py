#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:06:57 2019

@author: YuxuanLong


This script includes basic functions for:
    1. Data loader
    2. Model loader and loss loader
    3. Data augmentation and transformation
    4. Converter between Numpy and Torch tensor


"""

import os
import numpy as np
import cv2

from skimage import io, transform
import torch
from torch.autograd import Variable
import torchvision.utils

import torch.utils.data as utils_data

import model


# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)

def np_to_var(x):
    '''
    Converts numpy to Torch variable.
    '''
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    '''
    Converts Torch variable to numpy.
    '''
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def normalize_both(image, mask):
    image = np.array(image).astype(np.float32) / 255.0
    mask = np.array(mask).astype(np.float32) / 255.0
    return image, mask
    
def image_resize(image, resize, size):
    if resize:
        image = transform.resize(image, (size, size), mode = 'constant', anti_aliasing = True)
    image = np.moveaxis(image, 2, 0).astype(np.float32) # tensor format
    return image

def mask_resize(mask, resize, size):
    
    if resize:
        mask = transform.resize(mask, (size, size), mode = 'constant', anti_aliasing = True)
    mask = np.array(mask >= 0.5).astype(np.float32) # test here
    mask = np.expand_dims(mask, axis = 0)
    return mask

def rotate_both(image, mask):
    if np.random.random() < 0.5:
        image = np.rot90(image, 1, (0, 1))
        mask = np.rot90(mask, 1)
    return image, mask


def random_color(image, delta_h = 10, delta_s = 10, delta_v = 10):
    
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    u1 = np.random.random() * 2 - 1
    u2 = np.random.random() * 2 - 1
    u3 = np.random.random() * 2 - 1
    h = cv2.add(h, round(u1 * delta_h))
    s = cv2.add(s, round(u2 * delta_s))
    v = cv2.add(v, round(u3 * delta_v))
    out = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    
    return out
    
def random_rotate(image, mask, s):
    L = image.shape[0]
    C = np.array([L - 1, L - 1]) / 2.0
    
    
    theta = np.random.random() * np.pi / 2.0

    C_hat = np.array([s - 1, s - 1]) / 2.0
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    R = np.array([[cos, sin],[-sin, cos]])
    M = np.column_stack((R, C - np.dot(R, C_hat)))
    
    image = cv2.warpAffine(image, M, (s, s), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (s, s), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
    return image, mask

def flip_both(image, mask):
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if np.random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask


def flip_rotate(image, flip_v, flip_h, rotate, inverse = False):
    if inverse:
        # flip_v, flip_h, rotate
        if flip_v:
            image = np.flipud(image)   
        if flip_h:
            image = np.fliplr(image)  
        if rotate:
            image = np.rot90(image, 3, (0, 1))          
    else:
        # rotate, flip_h, flip_v
        if rotate:
            image = np.rot90(image, 1, (0, 1))  
        if flip_h:
            image = np.fliplr(image)
        if flip_v:
            image = np.flipud(image)
    return image
       

class MyDataset(utils_data.Dataset):
    def __init__(self, root, resize, data_augment, size, rotate, change_color):
        self.size = size
        self.root = root
        self.rotate = rotate
        self.data_augment = data_augment
        self.change_color = change_color
        mask_dir = root + '/groundtruth'
        self.resize = resize
        self.mask_file_list = [f for f in os.listdir(mask_dir) if 'sat' in f and 'png' in f]
#        random.shuffle(self.mask_file_list)

    def __getitem__(self, index):
        file_name =  self.mask_file_list[index].rsplit('.', 1)[0]
        img_name = self.root + '/images/' + file_name+'.png'
        mask_name = self.root + '/groundtruth/' + file_name+'.png'
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        if self.change_color:
            image = random_color(image)
        
        image, mask = normalize_both(image, mask)
        
        if self.rotate:
            self.resize = False
            image, mask = random_rotate(image, mask, self.size)
        
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
    


def get_data_loader(root, resize = True, data_augment = True,
                    image_size = 384, batch_size=100, rotate = False, change_color = False):
    """Creates training data loader."""
    train_dataset = MyDataset(root, resize, data_augment, image_size, rotate, change_color)
    return utils_data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)


def create_models(model_choice = 0):
    if model_choice == 1:
        net = model.D_LinkNet()
    elif model_choice == 0:
        net = model.LinkNet()
    elif model_choice == 2:
        net = model.D_LinkNetPlus()
    else:
        net = model.LinkNet()

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net


def loss(smooth, lam, gamma, loss_type):
    return model.Loss(smooth, lam, gamma, loss_type)
