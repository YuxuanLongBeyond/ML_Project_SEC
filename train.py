#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:07:35 2019

@author: YuxuanLong
"""

import os
import shutil
import numpy as np
import random
import time
import torch
import torch.optim as optim
import cv2
from skimage import io
import utils
import test

from torch import autograd


"""
This script is for training the neural network.
"""




# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

SEED = 2019
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    data_augment = True
    rotate = True
    change_color = True
    early_stop = False
    lr_decay = False
    model_choice = 2 # 0 for linknet, 1 Dlinknet, 2 for D_plusNet

    image_size = 384
    batch_size = 20
    num_epochs = 1500
    save_test_image = 10
    
    test_image_name = './data/test_set_images/test_26/test_26.png'
    validate_root = './data/validate'
    
    early_stop_tol = 8
    save_ckpt = 20
    
    lr = 1e-4
    decay_rate = 0.6
    decay_period = 500
    
    
    weight_decay = 1e-5
    
    
    loss_type = 'bce'    
    
    # BCE loss
    smooth = 1.0
    lam = 1.0
    
    # Focal loss
    gamma = 2.0
    

    root = './data/training'
    resize = True

    if os.path.exists('./epoch_output'):
        shutil.rmtree('./epoch_output')
    os.makedirs('./epoch_output')
    
    if not os.path.exists('./parameters'):
        os.makedirs('./parameters')    
    
    
    net = utils.create_models(model_choice)
    net.train() # in train mode
    
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    
    # create optimizers
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay, amsgrad = True)
    Loss = utils.loss(smooth, lam, gamma, loss_type)


    dataloader = utils.get_data_loader(root, resize, data_augment, image_size, batch_size, rotate, change_color)

    num_batch = len(dataloader)
    total_train_iters = num_epochs * num_batch

    loss_history = []
    print('Started training at {}'.format(time.asctime(time.localtime(time.time()))))
    test_loss = 100.0
    count = 0
    for epoch in range(num_epochs):
        print('Start epoch ', epoch)
        epoch_loss = 0 
        t = time.time()
        for iteration, batch in enumerate(dataloader, epoch * num_batch + 1):
            print('Iteration: ', iteration)
            print('Time for loading the data takes: ', time.time() - t, ' s')
            t = time.time()
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
            print('Time spent on back propagation: ', time.time() - t, ' s')
            loss_history.append(loss.data.item())
            t = time.time()
            
        # save the test image for visualizing the training outcome
        if (epoch + 1) % save_test_image == 0:
            with torch.no_grad():
                _, test_image = test.test_single_image(net, test_image_name, resize = False)  
            io.imsave('./epoch_output/test_epoch' + str(epoch) + '.png', test_image)
        
        if early_stop and (epoch + 1) % save_ckpt == 0:
            with torch.no_grad():
                loss, f1 = test.test_batch_with_labels(net, validate_root, batch_size = 10, image_size = 384, smooth = 1.0, lam = 1.0)
                print('On the validation dataset, loss: ', loss, ', F1: ', f1)
                if loss <= test_loss:
                    test_loss = loss
                    count = 0
                    torch.save(net.state_dict(), './parameters/weights')
                elif count < early_stop_tol:
                    count += 1
                    lr *= decay_rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr                
                    print('The new loss is found to be larger than before')
                else:
                    print('Reach the early stop tolerence...')
                    print('Break the update at ', epoch, 'th epoch')
                    break
        
        if not early_stop and (epoch + 1) % save_ckpt == 0:
            with torch.no_grad():
                torch.save(net.state_dict(), './parameters/weights')
              
        if lr_decay and (epoch + 1) % decay_period == 0: 
            with torch.no_grad():
                lr *= decay_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr     
        
        
        epoch_loss /= num_batch
        print('In the epoch ', epoch, ', the average batch loss is ', epoch_loss)
#    torch.save(net.state_dict(), './parameters/weights')
        
    # save the loss history
    with open('loss.txt', 'wt') as file:
        file.write('\n'.join(['{}'.format(loss) for loss in loss_history]))
        file.write('\n')