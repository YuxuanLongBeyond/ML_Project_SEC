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
    Convert numpy array to Torch variable.
    '''
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    '''
    Convert Torch variable to numpy array.
    '''
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def normalize_both(image, mask):
    '''
    Normalize the image and mask into range [0, 1].
    '''
    image = np.array(image).astype(np.float32) / 255.0
    mask = np.array(mask).astype(np.float32) / 255.0
    return image, mask
    
def image_resize(image, resize, size):
    '''
    Optionally resize the image into (size, size, 3).
    Then convert it into tensor format (3, size, size).
    '''
    if resize:
        image = transform.resize(image, (size, size), mode = 'constant', anti_aliasing = True)
    image = np.moveaxis(image, 2, 0).astype(np.float32) # tensor format
    return image

def mask_resize(mask, resize, size):
    '''
    Optionally resize the mask into (size, size), transform it to binary mask.
    Then convert it into tensor format (1, size, size).
    '''
    if resize:
        mask = transform.resize(mask, (size, size), mode = 'constant', anti_aliasing = True)
    mask = np.array(mask >= 0.5).astype(np.float32) # test here
    mask = np.expand_dims(mask, axis = 0)
    return mask

def rotate_both(image, mask):
    '''
    For a half probability, we rotate the image and mask by both 90 degrees.
    '''
    if np.random.random() < 0.5:
        image = np.rot90(image, 1, (0, 1))
        mask = np.rot90(mask, 1)
    return image, mask


def random_color(image, delta_h = 10, delta_s = 10, delta_v = 10):
    '''
    Add random perturbation on HSV channels of the image.

    Parameters:
        @delta_h: the range of perturbation on H channel (Hue).
        @delta_s: the range of perturbation on S channel (Saturation).
        @delta_v: the range of perturbation on V channel (Value).
    '''    
    
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
    
def random_rotate(image, mask, size):
    '''
    Rotate the image and mask by some random angle between 0 and 90 degrees.
    '''
    L = image.shape[0]
    C = np.array([L - 1, L - 1]) / 2.0
    
    theta = np.random.random() * np.pi / 2.0

    C_hat = np.array([size - 1, size - 1]) / 2.0
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    R = np.array([[cos, sin],[-sin, cos]]) # rotation matrix
    M = np.column_stack((R, C - np.dot(R, C_hat))) # affine transformation matrix
    
    # inverse mapping with cubic interpolation
    image = cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (size, size), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
    return image, mask

def flip_both(image, mask):
    '''
    With probability 0.5, the image and mask are flipped horizontally.
    With another probability 0.5, the image and mask are flipped vertically.
    '''
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if np.random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask


def flip_rotate(image, flip_v, flip_h, rotate, inverse = False):
    '''
    Flip and rotate image based on the flags.
    The operations can be inversed.
    This function is used for test time augmentation
    '''
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
    '''
    Task-oriented Torch dataset mainly for training, including data augmentation
    '''
    def __init__(self, root, resize, data_augment, size, rotate, change_color):
        '''
        Parameters:
            @root: the root directory for the images and masks.
            @resize: boolean flag for resize.
            @data augment: boolean flag for random flip and 90-degree rotation (DA8).
            @size: size of image and mask to be trained or validated.
            @rotate: boolean flag for random rotation.
            @change_color: boolean flag for random perturbation on HSV channels.
        '''
        
        self.size = size
        self.root = root
        self.rotate = rotate
        self.data_augment = data_augment
        self.change_color = change_color
        mask_dir = root + '/groundtruth'
        self.resize = resize
        self.mask_file_list = [f for f in os.listdir(mask_dir) if 'sat' in f and 'png' in f]

    def __getitem__(self, index):
        '''
        Given an index in file list, we extract the corresponding image and mask.
        Data augmentation is then applied on image and mask.
        Finally, a dictionary for image and mask is returned.
        '''
        file_name =  self.mask_file_list[index].rsplit('.', 1)[0] # image name without .png
        img_name = self.root + '/images/' + file_name+'.png'
        mask_name = self.root + '/groundtruth/' + file_name+'.png'
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        # HSV random perturbation
        if self.change_color:
            image = random_color(image)
        
        image, mask = normalize_both(image, mask)
        
        # Random rotation
        if self.rotate:
            self.resize = False
            image, mask = random_rotate(image, mask, self.size)
        
        # DA8
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
                    image_size = 384, batch_size=20, rotate = False, change_color = False):
    '''
    Get the data loader from my dataset.
    Parameters:
        @root: the root directory for the images and masks.
        @resize: boolean flag for image resize.
        @data augment: boolean flag for random flip and 90-degree rotation (DA8).
        @image_size: size of image and mask to be trained or validated.
        @batch_size: batch size during training or validation.
        @rotate: boolean flag for random rotation.
        @change_color: boolean flag for random perturbation on HSV channels.    
    '''
    train_dataset = MyDataset(root, resize, data_augment, image_size, rotate, change_color)
    return utils_data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)


def create_models(model_choice = 2):
    '''
    Choose one model from our implementations
    '''
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
    '''
    Parameters:
        @smooth: number to be added on denominator and numerator when compute dice loss.
        @lam: weight to balance the dice loss in the final combined loss.
        @gamma: for focal loss.
        @loss_type: 'bce' or 'focal'.
    Return: object for combined loss
    '''
    return model.Loss(smooth, lam, gamma, loss_type)
