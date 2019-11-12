#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:09:53 2019

@author: YuxuanLong
"""
import os
from skimage import io, transform
import numpy as np

def crop_both(image, mask, index, i_s, fixed_size = 1280, step = 400):
    w, h, _ = image.shape
    
#    crop_size = low_size + i_s * step
    crop_size = fixed_size + (i_s + 1) * step
    
    if index == 0:
        # left upper
        image = image[0:fixed_size, 0:fixed_size, :]
        mask = mask[0:fixed_size, 0:fixed_size]
    elif index == 1:
        # right upper
        image = image[0:fixed_size, (h - fixed_size):, :]
        mask = mask[0:fixed_size, (h - fixed_size):]
    elif index == 2:
        # right bottom
        image = image[(w - fixed_size):, (h - fixed_size):, :]
        mask = mask[(w - fixed_size):, (h - fixed_size):]
    elif index == 3:
        # left bottom
        image = image[(w - fixed_size):, 0:fixed_size, :]
        mask = mask[(w - fixed_size):, 0:fixed_size]
    else:
        c_y = w // 2
        c_x = h // 2
        rand_size = crop_size // 2
        image = image[(c_y - rand_size):(c_y + rand_size), (c_x - rand_size):(c_x + rand_size), :]
        mask = mask[(c_y - rand_size):(c_y + rand_size), (c_x - rand_size):(c_x + rand_size)]
    return image, mask

def crop_from_center(image, mask, crop_size = 2560):
    w, h, _ = image.shape
    c_y = w // 2
    c_x = h // 2
    rand_size = crop_size // 2
    image = image[(c_y - rand_size):(c_y + rand_size), (c_x - rand_size):(c_x + rand_size), :]
    mask = mask[(c_y - rand_size):(c_y + rand_size), (c_x - rand_size):(c_x + rand_size)]
    return image, mask

def crop_from_ones(image, mask, crop_size = 2560):
    w, h, _ = image.shape
    c_y = w // 2
    c_x = h // 2
    rand_size = crop_size // 2

    mask_c = mask[(c_y - rand_size):(c_y + rand_size), (c_x - rand_size):(c_x + rand_size)]
    
    mask_l = mask[:, 0:crop_size]
    
    mask_r = mask[:, (h - crop_size):]
    
    return image, mask

if __name__ == '__main__':
    root = './data/chicago'
    thresh = 0.01
    

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
    file_list = [[image_file_list[i], label_file_list[i]] for i in range(file_num)]  
    
    out_size = 608
    num = 50
    count = 1
    for pair in file_list:

        
        img_name = root + '/' + pair[0]
        mask_name = root + '/' + pair[1]
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        
        mask = (mask[:, :, 0] == 0).astype(np.float32)        
        
        image = np.array(image).astype(np.float32) / 255.0
        
        
#        for j in range(4):
#            k = 0
#            image_hat, mask_hat = crop_both(image, mask, j, k)
#                
#                
#            ratio = np.sum(mask_hat) / (mask_hat.shape[0] * mask_hat.shape[1])
#            if ratio >= thresh:
#                image_hat = transform.resize(image_hat, (out_size, out_size), mode = 'constant', anti_aliasing=True)
#                mask_hat = transform.resize(mask_hat, (out_size, out_size), mode = 'constant', anti_aliasing=True)
#                name = 'sat%d_%d_%d.png' % (count, j, k)
#                io.imsave('./data/my_data/images/' + name, image_hat)
#                io.imsave('./data/my_data/groundtruth/' + name, mask_hat)
#                
#        for k in range(3):
#            j = 4
#            image_hat, mask_hat = crop_both(image, mask, j, k)
#                
#                
#            ratio = np.sum(mask_hat) / (mask_hat.shape[0] * mask_hat.shape[1])
#            if ratio >= thresh:
#                image_hat = transform.resize(image_hat, (out_size, out_size), mode = 'constant', anti_aliasing=True)
#                mask_hat = transform.resize(mask_hat, (out_size, out_size), mode = 'constant', anti_aliasing=True)
#                name = 'sat%d_%d_%d.png' % (count, j, k)
#                io.imsave('./data/my_data/images/' + name, image_hat)
#                io.imsave('./data/my_data/groundtruth/' + name, mask_hat)
        image_hat, mask_hat = crop_from_center(image, mask)
        image_hat = transform.resize(image_hat, (out_size, out_size), mode = 'constant', anti_aliasing=True)
        mask_hat = transform.resize(mask_hat, (out_size, out_size), mode = 'constant', anti_aliasing=True)
        name = 'sat%d.png' % count
        io.imsave('./data/my_data/images/' + name, image_hat)
        io.imsave('./data/my_data/groundtruth/' + name, mask_hat)
        count += 1