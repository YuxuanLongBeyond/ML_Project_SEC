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
        rand_size = round(np.random.random() * high_size / 2.0) - 1
        image = image[(c_y - rand_size):(c_y + rand_size), (c_x - rand_size):(c_x + rand_size), :]
    return image, mask

if __name__ == '__main__':
    root = './data/test_only'
    tresh = 0.001
    

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
    
    