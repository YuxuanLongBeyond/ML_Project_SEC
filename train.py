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
import numpy as np
import random
import time
from skimage import io
import torch
import torch.optim as optim

import utils
import test

from torch import autograd


# Run on GPU if CUDA is available.
RUN_ON_GPU = torch.cuda.is_available()

SEED = 2019
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    new_data = True
    data_augment = True
    

    image_size = 384
    batch_size = 50
    num_epochs = 60
    save_interval = 1
    save_ckpt = 1
    test_image_name = './data/main_data/test_set_images/test_26/test_26.png'
    
    lr = 1e-4 # 2e-4
    weight_decay = 1e-5
    smooth = 1.0
    lam = 1.0
    
    
    if new_data:
        root = '../my_data'
        resize = False
    else:
        root = './data/main_data/training'
        resize = True
     
    if os.path.exists('./epoch_output'):
        shutil.rmtree('./epoch_output')
    os.makedirs('./epoch_output')
    
    if not os.path.exists('./parameters'):
        os.makedirs('./parameters')    
    
    
    net = utils.create_models()
    net.train()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    # create optimizers
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay, amsgrad = True)
    Loss = utils.loss(smooth, lam)


    dataloader = utils.get_data_loader(root, False, resize, data_augment, image_size, batch_size)
    num_batch = len(dataloader)
    total_train_iters = num_epochs * num_batch

    loss_history = []
    print('Started training at {}'.format(time.asctime(time.localtime(time.time()))))
    
    for epoch in range(num_epochs):
        print('Start epoch ', epoch)
        epoch_loss = 0 
        for iteration, batch in enumerate(dataloader, epoch * num_batch + 1):
            t = time.time()
            print('Iteration: ', iteration)
            image = utils.np_to_var(batch['image'])
            mask = utils.np_to_var(batch['mask'])
            

            optimizer.zero_grad()

            pred = net.forward(image)

            
            loss = Loss.final_loss(pred, mask)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data.item()

            # print the log info
            print('Iteration [{:6d}/{:6d}] | loss: {:.4f}'.format(
                iteration, total_train_iters, loss.data.item()))
            print('Time spent: ', time.time() - t, ' s')
            # keep track of loss for plotting and saving
        if (epoch + 1) % save_interval == 0:
            with torch.no_grad():
                _, test_image = test.test_single_image(net, test_image_name, resize = False)  
                io.imsave('./epoch_output/test_epoch' + str(epoch) + '.png', test_image)
        
        if (epoch + 1) % save_ckpt == 0:
            torch.save(net.state_dict(), './parameters/weights')
        
        epoch_loss /= num_batch
        print('In the epoch ', epoch, ', the average loss is ', epoch_loss)
        
        
    torch.save(net.state_dict(), './parameters/weights')
    # save the loss history
    with open('loss.txt', 'wt') as file:
        file.write('\n'.join(['{}'.format(loss) for loss in loss_history]))
        file.write('\n')