#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:07:35 2019

@author: YuxuanLong
"""

# Loss, train


import os
# import argparse
import random
import numpy as np
import time

from skimage import io, transform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils

import torch.utils.data as utils_data

import model


# Run on GPU if CUDA is available.
RUN_ON_GPU = torch.cuda.is_available()

SEED = 0
ROOT = './data/main_data/training'

np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


def np_to_var(x):
    """Converts numpy to variable."""
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    """Converts variable to numpy."""
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def image_transform(image, resize, size):
    if resize:
        image = transform.resize(image, (size, size), mode = 'constant', anti_aliasing=True)
    image = np.array(image)
    image = np.moveaxis(image, 2, 0) # tensor format
    image = image.astype(np.float32)
    return image
        
def mask_transform(mask, resize, size):
    if resize:
        mask = transform.resize(mask, (size, size), mode = 'constant', anti_aliasing=True)
    mask = np.array(mask >= 0.5).astype(np.float32) # test here
    mask = np.expand_dims(mask, axis = 0)
    return mask

class MyDataset(utils_data.Dataset):
    def __init__(self, resize = None, size = 384):
        self.size = size
        mask_dir = ROOT + '/groundtruth'
        self.resize = resize
        self.mask_file_list = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
        random.shuffle(self.mask_file_list)

    def __getitem__(self, index):
        file_name =  self.mask_file_list[index].rsplit('.', 1)[0]
        img_name = ROOT + '/images/' + file_name+'.png'
        mask_name = ROOT + '/groundtruth/' + file_name+'.png'
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        image = image_transform(image, self.resize, self.size)
        mask = mask_transform(mask, self.resize, self.size)
        
        sample = {'image': image, 'mask': mask}

        return sample
  
    def __len__(self):
        return len(self.mask_file_list)


def get_data_loader(resize = True, image_size = 384, batch_size=10):
    """Creates training data loader."""
    train_dataset = MyDataset(resize, image_size)
    return utils_data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)


def create_models():

    net = model.LinkNet()

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net



    
    

if __name__ == '__main__':
    net = create_models()
    image_size = 384
    batch_size = 20
    num_epochs = 1
    save_interval = 1
    out_dir = 'output'
    test_image_name = ROOT + '/images/satImage_001.png'
    resize = True
    
    lr = 2e-4
    weight_decay = 1e-5
    smooth = 1.0
    lam = 1.0
    beta = 0.7
    
    ##create single image tensor for test in each epoch
    test_image_origin = io.imread(test_image_name)
    test_image_origin = image_transform(test_image_origin, resize, image_size)
    test_image_dum = np.moveaxis(test_image_origin, 0, 2)
    test_image = np.expand_dims(test_image_origin, axis = 0)
    test_image = np_to_var(torch.from_numpy(test_image))    
    
    
    
    # create optimizers
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay, amsgrad = True)
    Loss = model.Loss(smooth, lam, beta)
    # create output directory for image samples
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataloader = get_data_loader(resize, image_size = image_size, batch_size = batch_size)
    num_batch = len(dataloader)
    total_train_iters = num_epochs * num_batch

    loss_history = []
    print('Started training at {}'.format(time.asctime(time.localtime(time.time()))))
    
    for epoch in range(num_epochs):
        print('Start epoch ', epoch)
        epoch_loss = 0 
        for iteration, batch in enumerate(dataloader, epoch * num_batch + 1):
            print('Iteration: ', iteration)
            image = np_to_var(batch['image'])
            mask = np_to_var(batch['mask'])
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
            loss_history.append(loss.data.item())
#
##            # save the generated samples and loss
#            if iteration % save_interval == 0:
#
#                # save the loss history
#                with open('loss.txt', 'wt') as file:
#                    file.write('\n'.join(['{}'.format(loss) for loss in loss_history]))
#                    file.write('\n')
            
        ###dummy test
        pred_test = net.forward(test_image)
        pred_np = var_to_np(pred_test)[0][0]
        
        new_mask = pred_np >= 0.5
        
        dummy = test_image_dum + 0
        
        channel = dummy[:, :, 0]
        channel[new_mask] = 1.0
        dummy[:, :, 0] = channel
            
        dummy = (dummy * 255).astype(np.uint8)
        io.imsave('./epoch_output/test_output_iter' + str(iteration) + '.png', dummy, quality = 100)
                    
        epoch_loss /= num_batch
        print('In the epoch ', epoch, ', the average loss is ', epoch_loss)
        
        
        
        
        

