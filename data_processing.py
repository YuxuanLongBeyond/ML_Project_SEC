#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:09:53 2019

@author: YuxuanLong
"""
import os
from skimage import io, transform

if __name__ == '__main__':
    root = './data/test_only'
    
    

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
    
    