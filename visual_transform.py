#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:54:00 2019

@author: YuxuanLong
"""
import numpy as np
import cv2
from skimage import io, transform

####### Visualizing the geometric and color transformation

def random_color(image, delta_h = 0, delta_s = 0, delta_v = 0):
    
    tem = image[:, :, 2]
    image[:, :, 2] = image[:, :, 0]
    image[:, :, 0] = tem
    
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    u1 = np.random.random() * 2 - 1
    u2 = np.random.random() * 2 - 1
    u3 = np.random.random() * 2 - 1
    h = cv2.add(h, round(u1 * delta_h))
    s = cv2.add(s, round(u2 * delta_s))
    v = cv2.add(v, round(u3 * delta_v))
    out = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    
    tem = out[:, :, 2]
    out[:, :, 2] = out[:, :, 0]
    out[:, :, 0] = tem
    return out
    
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
    