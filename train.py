#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:07:35 2019

@author: YuxuanLong
"""

# Loss, train


import os
import argparse
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


# --- utility code -------------------------------------------------------------

# Run on GPU if CUDA is available.
RUN_ON_GPU = torch.cuda.is_available()

# Set the random seed for reproducibility
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

class MyDataset(utils_data.Dataset):
    def __init__(self, transform = None, size = 384):
        self.size = size
        self.dataset_path = ROOT
        self.image_dir = ROOT + '/images'
        self.mask_dir = ROOT + '/groundtruth'
        self.transform = transform
        self.mask_file_list = [f for f in os.listdir(self.mask_dir) if os.path.isfile(os.path.join(self.mask_dir, f))]
        random.shuffle(self.mask_file_list)

    def __getitem__(self, index):
        file_name =  self.mask_file_list[index].rsplit('.', 1)[0]
        img_name = self.dataset_path + '/images/' + file_name+'.png'
        mask_name = self.dataset_path + '/groundtruth/' + file_name+'.png'
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        if self.transform:
            image = transform.resize(image, (self.size, self.size), mode = 'constant', anti_aliasing=True)
            mask = transform.resize(mask, (self.size, self.size), mode = 'constant', anti_aliasing=True)
            
        image = np.array(image)
        image = np.rollaxis(image, axis = 2, start = 0) # tensor format
        image = np.array(image).astype(np.float32)
        
        
        mask = np.array(mask >= 128).astype(np.float32)
        mask = np.expand_dims(mask, axis = 0)
        
        sample = {'image': image, 'mask': mask}

        return sample
  
    def __len__(self):
        return len(self.mask_file_list)


def get_data_loader(image_size = 384, batch_size=10):
    """Creates training data loader."""

    train_dataset = MyDataset(True, image_size)
    return utils_data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)

# --- training code -------------------------------------------------------

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
    batch_size = 10
    num_epochs = 1
    save_interval=1
    out_dir='output'
    
    lr = 2e-4
    weight_decay = 1e-5
    
    # create optimizers
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay, amsgrad = True)

    # create output directory for image samples
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataloader = get_data_loader(image_size = image_size, batch_size = batch_size)
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

            loss = nn.BCELoss()
            loss = loss(torch.sigmoid(pred), mask)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data.item()

            # print the log info
            print('Iteration [{:6d}/{:6d}] | loss: {:.4f}'.format(
                iteration, total_train_iters, loss.data.item()))
            
            # keep track of loss for plotting and saving
            loss_history.append(loss.data.item())
#
#            # save the generated samples and loss
            if iteration % save_interval == 0:

                # save the loss history
                with open('loss.txt', 'wt') as file:
                    file.write('\n'.join(['{}'.format(loss) for loss in loss_history]))
                    file.write('\n')
        epoch_loss /= num_batch

