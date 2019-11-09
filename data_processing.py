#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:09:53 2019

@author: YuxuanLong
"""
import os
from skimage import io, transform
import numpy as np

def crop_both(image, mask, low_size = 608, high_size = 2560):
    w, h, _ = image.shape
    
    crop_size = low_size + np.random.random() * (high_size - low_size)
    crop_size = round(crop_size)
#        r1 = np.random.random()
    r2 = np.random.random()
    
    if r2 < 0.2:
        # left upper
        image = image[0:crop_size, 0:crop_size, :]
        mask = mask[0:crop_size, 0:crop_size]
    elif r2 < 0.4:
        # right upper
        image = image[0:crop_size, (h - crop_size):, :]
        mask = mask[0:crop_size, (h - crop_size):]
    elif r2 < 0.6:
        # right bottom
        image = image[(w - crop_size):, (h - crop_size):, :]
        mask = mask[(w - crop_size):, (h - crop_size):]
    elif r2 < 0.8:
        # left bottom
        image = image[(w - crop_size):, 0:crop_size, :]
        mask = mask[(w - crop_size):, 0:crop_size]
    else:
        c_y = w // 2
        c_x = h // 2
        rand_size = round(np.random.random() * (high_size - low_size) / 2.0 + low_size / 2)
        image = image[(c_y - rand_size):(c_y + rand_size), (c_x - rand_size):(c_x + rand_size), :]
    return image, mask

if __name__ == '__main__':
    root = './data/test_only'
    thresh = 0.001
    

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
    
    out_size = 384
    num = 50
    count = 1
    for pair in file_list:

        
        img_name = root + '/' + pair[0]
        mask_name = root + '/' + pair[1]
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        
        mask = (mask[:, :, 0] == 0).astype(np.float32)        
        
        image = np.array(image).astype(np.float32) / 255.0
        
        
        for i in range(num):
            image_hat, mask_hat = crop_both(image, mask)
            
            ratio = np.sum(mask_hat) / (mask_hat.shape[0] * mask_hat.shape[1])
            if ratio >= thresh:
                image_hat = transform.resize(image_hat, (out_size, out_size), mode = 'constant', anti_aliasing=True)
                mask_hat = transform.resize(mask_hat, (out_size, out_size), mode = 'constant', anti_aliasing=True)
                name = 'sat' + str(count) + '_' + str(i) + '.png'
                io.imsave('./data/my_data/train/' + name, image_hat)
                io.imsave('./data/my_data/ground_truth/' + name, mask_hat)
        
        count += 1