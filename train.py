#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:07:35 2019

@author: YuxuanLong
"""

# Loss, train


import os
import shutil
# import argparse
from skimage import io
import numpy as np
import time

import torch
import torch.optim as optim

import utils


# Run on GPU if CUDA is available.
RUN_ON_GPU = torch.cuda.is_available()

SEED = 0

np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    root = './data/main_data/training'
    if os.path.exists('./epoch_output'):
        shutil.rmtree('./epoch_output')
    os.makedirs('./epoch_output')
    net = utils.create_models()
    image_size = 384
    batch_size = 1
    num_epochs = 1
    save_interval = 1

    test_image_name = root + '/images/satImage_001.png'
    resize = True
    
    lr = 2e-4
    weight_decay = 1e-5
    smooth = 1.0
    lam = 1.0
    beta = 0.5
    
    ##create single image tensor for test in each epoch
    test_image_origin = io.imread(test_image_name)
    test_image_origin = utils.image_transform(test_image_origin, resize, image_size)
    test_image_dum = np.moveaxis(test_image_origin, 0, 2)
    test_image = np.expand_dims(test_image_origin, axis = 0)
    test_image = utils.np_to_var(torch.from_numpy(test_image))    
    
    
    
    # create optimizers
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay, amsgrad = True)
    Loss = utils.loss(smooth, lam, beta)


    dataloader = utils.get_data_loader(root, resize, image_size = image_size, batch_size = batch_size)
    num_batch = len(dataloader)
    total_train_iters = num_epochs * num_batch

    loss_history = []
    print('Started training at {}'.format(time.asctime(time.localtime(time.time()))))
    
    for epoch in range(num_epochs):
        print('Start epoch ', epoch)
        epoch_loss = 0 
        for iteration, batch in enumerate(dataloader, epoch * num_batch + 1):
            print('Iteration: ', iteration)
            image = utils.np_to_var(batch['image'])
            mask = utils.np_to_var(batch['mask'])
#
            optimizer.zero_grad()

            pred = net.forward(image)

            
            loss = Loss.final_loss(pred, mask)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data.item()

            # print the log info
            print('Iteration [{:6d}/{:6d}] | loss: {:.4f}'.format(
                iteration, total_train_iters, loss.data.item()))
            
            # keep track of loss for plotting and saving

            
        ###dummy test
        pred_test = net.forward(test_image)
        pred_np = utils.var_to_np(pred_test)[0][0]
        
        new_mask = pred_np >= 0.5
        
        dummy = test_image_dum + 0
        
        channel = dummy[:, :, 0]
        channel[new_mask] = 1.0
        dummy[:, :, 0] = channel
            
        dummy = (dummy * 255).astype(np.uint8)
        io.imsave('./epoch_output/test_output_iter' + str(iteration) + '.png', dummy)
                    
        epoch_loss /= num_batch
        print('In the epoch ', epoch, ', the average loss is ', epoch_loss)
        
        
    torch.save(net.state_dict(), './parameters/weights')
    # save the loss history
    with open('loss.txt', 'wt') as file:
        file.write('\n'.join(['{}'.format(loss) for loss in loss_history]))
        file.write('\n')

