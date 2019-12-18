#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:54:00 2019

@author: YuxuanLong
"""
import numpy as np
import cv2
from skimage import io, transform
from utils import random_color

####### Visualizing the geometric and color transformation
### mainly for debug test of data augmentation
    
if __name__ == '__main__':
    A = io.imread('./data/training/images/satImage_001.png')
    A = random_color(A)
    A = np.array(A).astype(np.float32) / 255.0

    
    L = A.shape[0]
    C = np.array([L - 1, L - 1]) / 2.0
    
    
    theta = np.pi / 6

    C_hat = C
    s = 400
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    R = np.array([[cos, sin],[-sin, cos]])
    M = np.column_stack((R, C - np.dot(R, C_hat)))
    
    B = cv2.warpAffine(A, M, (s, s), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT_101)
    
    B = np.rot90(B, 1, (0, 1))
    
    B = transform.resize(B, (384, 384), mode = 'constant', anti_aliasing = True)
    
    io.imshow(B)
    io.imsave('rotate_30.png', B)
    