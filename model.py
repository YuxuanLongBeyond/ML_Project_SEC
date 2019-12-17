#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:23:39 2019

@author: YuxuanLong

The following implements three network models in Pytorch framework:
    1. LinkNet
    2. D-LinkNet
    3. D-LinkNet+

ResNet34 is used as the encoder to all the models.

We denote that:
    N--batch size
    H--input image height
    W--input image width

"""


import torch
import torch.nn as nn
from torchvision import models

class LinkNet(nn.Module):
    '''
    LinkNet
    '''
    def __init__(self):
        
        # subclass of nn.Module
        super(LinkNet, self).__init__()
        
        resnet = models.resnet34(pretrained = True)


        layer0 = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool]
        self.layer0 = nn.Sequential(*layer0)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder1 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder3 = Decoder(128, 64)
        self.decoder4 = Decoder(64, 64)
        
        decoder5 = [nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1), 
                    nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, kernel_size = 3, padding = 1), 
                    nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, kernel_size = 3, padding = 1)]
        self.decoder5 = nn.Sequential(*decoder5)
        
        
    def forward(self, x):
        '''
        Parameters:
            @x: input image batch with size N * 3 * H * W
        
        return: confidence map (batch) with size N * 1 * H * W
        '''        
        x0 = self.layer0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        
        d1 = self.decoder1(x4) + x3
        d2 = self.decoder2(d1) + x2
        d3 = self.decoder3(d2) + x1
        d4 = self.decoder4(d3)
        out = self.decoder5(d4)

        return torch.sigmoid(out)

class D_LinkNet(nn.Module):
    '''
    D-LinkNet
    '''    
    def __init__(self):
        
        # subclass nn.Module
        super(D_LinkNet, self).__init__()
        
        resnet = models.resnet34(pretrained = True)


        layer0 = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool]
        self.layer0 = nn.Sequential(*layer0)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder1 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder3 = Decoder(128, 64)
        self.decoder4 = Decoder(64, 64)
        
        
        self.dblock = Dblock(512)
        decoder5 = [nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1), 
                    nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, kernel_size = 3, padding = 1), 
                    nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, kernel_size = 3, padding = 1)]
        self.decoder5 = nn.Sequential(*decoder5)
        
        
    def forward(self, x):
        '''
        Parameters:
            @x: input image batch with size N * 3 * H * W
        
        return: confidence map (batch) with size N * 1 * H * W
        '''            
        x0 = self.layer0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        x4 = self.dblock(x4)
        
        d1 = self.decoder1(x4) + x3
        d2 = self.decoder2(d1) + x2
        d3 = self.decoder3(d2) + x1
        d4 = self.decoder4(d3)
        out = self.decoder5(d4)

        return torch.sigmoid(out)
    

class D_LinkNetPlus(nn.Module):
    '''
    D-LinkNet+
    '''        
    def __init__(self):
        
        # subclass nn.Module
        super(D_LinkNetPlus, self).__init__()
        
        resnet = models.resnet34(pretrained = True)


        layer0 = [resnet.conv1, resnet.bn1, resnet.relu]
        self.layer0 = nn.Sequential(*layer0)
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder1 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder3 = Decoder(128, 64)
        self.decoder4 = Decoder(64, 64)
        
        
        self.dblock = Dblock(512)
        
        self.conv1 = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size = 3, padding = 1), nn.BatchNorm2d(256), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv2d(128, 128, kernel_size = 3, padding = 1), nn.BatchNorm2d(128), nn.ReLU()])
        self.conv3 = nn.Sequential(*[nn.Conv2d(64, 64, kernel_size = 3, padding = 1), nn.BatchNorm2d(64), nn.ReLU()])
        
        
        decoder5 = [nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1), 
                    nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, kernel_size = 3, padding = 1), 
                    nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, kernel_size = 3, padding = 1)]
        self.decoder5 = nn.Sequential(*decoder5)
        
        
    def forward(self, x):
        '''
        Parameters:
            @x: input image batch with size N * 3 * H * W
        
        return: confidence map (batch) with size N * 1 * H * W
        '''                
        x0 = self.layer0(x)
        x0_pool = self.maxpool(x0)
        x1 = self.encoder1(x0_pool)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        x4 = self.dblock(x4)
        
        d1 = self.decoder1(x4) + self.conv1(x3)
        d2 = self.decoder2(d1) + self.conv2(x2)
        d3 = self.decoder3(d2) + self.conv3(x1)
        d4 = self.decoder4(d3) # + self.conv4(x0)
        out = self.decoder5(d4)

        return torch.sigmoid(out)    
    

class Dblock(nn.Module):
    '''
    Cascading dilated convolutions between encoder and decoder
    (central part of D-LinkNet and D-LinkNet)
    '''
    def __init__(self, channel):
        '''
        Parameters:
            @channel: the number of channels from the output of the encoder
        '''
        super(Dblock, self).__init__()
        dilate1 = [nn.Conv2d(channel, channel, kernel_size = 3, dilation = 1, padding = 1), nn.ReLU()]
        dilate2 = [nn.Conv2d(channel, channel, kernel_size = 3, dilation = 2, padding = 2), nn.ReLU()]
        dilate3 = [nn.Conv2d(channel, channel, kernel_size = 3, dilation = 4, padding = 4), nn.ReLU()]
        
        self.dilate1 = nn.Sequential(*dilate1)
        self.dilate2 = nn.Sequential(*dilate2)
        self.dilate3 = nn.Sequential(*dilate3)
        
    def forward(self, x):
        '''
        Parameters:
            @x: latent features from encoder, with size N * channel * (H / 32) * (W / 32)
            
        return: features processed by cascaded dilated convolutions
        '''
        d1 = self.dilate1(x)
        d2 = self.dilate1(d1)
        d3 = self.dilate1(d2)
        out = x + d1 + d2 + d3
        return out
        
        
    
class Decoder(nn.Module):
    '''
    A decoder block that scale the size of input features by a factor of 2
    '''
    def __init__(self, c_in, c_out):
        '''
        Parameters:
            @c_in: the number of channels of the input features
            @c_out: the number of channels of the output features
        '''
        
        
        super(Decoder, self).__init__()
        
        tem = c_in // 4
        layer1 = [nn.Conv2d(c_in, tem, kernel_size = 1), nn.BatchNorm2d(tem), nn.ReLU()]
        layer2 = [nn.ConvTranspose2d(tem, tem, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), nn.BatchNorm2d(tem), nn.ReLU()]
        layer3 = [nn.ConvTranspose2d(tem, c_out, kernel_size = 1), nn.BatchNorm2d(c_out), nn.ReLU()]
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        
    def forward(self, x):
        '''
        Parameters:
            @x input to the decoder, with size N * C_in * s * s
        return: upsampled features with size N * C_out * (2s) * (2s)
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x        
    
class Loss(nn.Module):
    '''
    Define the loss
    '''
    def __init__(self, smooth, lam, gamma, loss_type = 'bce'):
        super(Loss, self).__init__()

        self.smooth = smooth
        self.lam = lam    
        self.gamma = gamma
        self.loss_type = loss_type
    
    def bce_loss(self, pred, mask):
        '''
        Standard BCE by Pytorch function
        '''
        loss = nn.BCELoss()
        return loss(pred, mask)
    
    def focal_loss(self, pred, mask, epsilon = 1e-6):
        '''
        Focal loss
        '''
        pred = torch.clamp(pred, min = epsilon, max = 1.0 - epsilon)
        loss = - mask * torch.pow(1.0 - pred, self.gamma) * torch.log(pred) - (1.0 - mask) * torch.pow(pred, self.gamma) * torch.log(1.0 - pred)
        return torch.mean(loss)
        
        
    def dice_loss(self, pred, mask):
        '''
        Dice loss
        '''
        # note that numer / denom is just F1 score
        numer = 2.0 * torch.sum(pred * mask, (1, 2, 3))
        denom = torch.sum(pred, (1, 2, 3)) + torch.sum(mask, (1, 2, 3))
        loss_batch = 1.0 - (numer + self.smooth) / (denom + self.smooth)
        return torch.mean(loss_batch)
        
    def final_loss(self, pred, mask):
        '''
        The final loss is either:
            1. BCE + dice loss
            2. focal loss + dice loss
        '''
        loss = self.dice_loss(pred, mask) * self.lam
        if self.loss_type == 'bce':
            loss += self.bce_loss(pred, mask)
        elif self.loss_type == 'focal':
            loss += self.focal_loss(pred, mask)
        else:
            raise ValueError
        
        return loss
        
