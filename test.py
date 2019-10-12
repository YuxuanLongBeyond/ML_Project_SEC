#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:10:29 2019

@author: YuxuanLong
"""
# show prediction during training

test_num = 0
pred_np = var_to_np(pred)[test_num][0]

new_mask = pred_np >= 0.5

image_up = var_to_np(image)[0]

image_new = np.moveaxis(image_up, 0, 2)

channel = image_new[:, :, 0]
channel[new_mask] = 1.0
image_new[:, :, 0] = channel

io.imshow(image_new)