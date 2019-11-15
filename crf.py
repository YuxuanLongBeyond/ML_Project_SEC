#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:29:49 2019

@author: YuxuanLong
"""

#full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)


import numpy as np
import pydensecrf.densecrf as dcrf
from skimage import io, transform
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]   
    
    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)
    
    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U).astype(np.float32)
    img = np.ascontiguousarray(img)
#
    d.setUnaryEnergy(U)
#
    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
#
    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w)).astype(np.float32)

    return Q

if __name__ == '__main__':
    test_image = io.imread('./data/main_data/test_set_images/test_1/test_1.png')
    
    test_mask = io.imread('./output/test1.png')
    
#    full_mask = dense_crf(test_image, test_mask)
    

#    img = np.moveaxis(img, 2, 0)
    
    
#    U = unary_from_labels(test_mask, 2, gt_prob=0.7)
    h = test_mask.shape[0]
    w = test_mask.shape[1]
    
    mask = test_mask > 0
    output_probs = np.zeros((h, w))
    output_probs[mask] = 0.9
    output_probs[~mask] = 0.1
    
    Q = dense_crf(test_image, output_probs)



#    