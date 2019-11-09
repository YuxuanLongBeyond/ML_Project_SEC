#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:23:39 2019

@author: YuxuanLong
"""

import torch
import torch.nn as nn
from torchvision import models

class LinkNet(nn.Module):
    # from Resnet34
    def __init__(self, fix_res = True):
        
        # subclass nn.Module
        super(LinkNet, self).__init__()
        
        resnet = models.resnet34(pretrained = True)
        
        for param in resnet.parameters():
            param.requires_grad = not fix_res

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
        
#        decoder5 = [nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1), 
#                    nn.ReLU(), nn.Conv2d(32, 32, kernel_size = 3, padding = 1), nn.ReLU(), 
#                    nn.Conv2d(32, 1, kernel_size = 3, padding = 1)]
        decoder5 = [nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1), 
                    nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, kernel_size = 3, padding = 1), 
                    nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, kernel_size = 3, padding = 1)]
        self.decoder5 = nn.Sequential(*decoder5)
        
        
    def forward(self, x):
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
    
class Decoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(Decoder, self).__init__()
        
        tem = c_in // 4
        layer1 = [nn.Conv2d(c_in, tem, kernel_size = 1), nn.BatchNorm2d(tem), nn.ReLU()]
        layer2 = [nn.ConvTranspose2d(tem, tem, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                       nn.BatchNorm2d(tem), nn.ReLU()]
        layer3 = [nn.ConvTranspose2d(tem, c_out, kernel_size = 1), nn.BatchNorm2d(c_out), nn.ReLU()]
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x        
    
class Loss(nn.Module):
    def __init__(self, smooth, lam, beta):
        super(Loss, self).__init__()

        self.smooth = smooth
        self.lam = lam
        self.beta = beta        
    
    def bce_loss(self, pred, mask):
        bce = - self.beta * mask * torch.log(pred) - (1 - self.beta) * (1 - mask) * torch.log(1 - pred)
        return torch.mean(bce)

    def dice_loss(self, pred, mask):
        
        numer = 2.0 * torch.sum(pred * mask, (1, 2, 3))
        denom = torch.sum(pred, (1, 2, 3)) + torch.sum(mask, (1, 2, 3))
        loss_batch = 1 - (numer + self.smooth) / (denom + self.smooth)
        return torch.mean(loss_batch)
        
    def final_loss(self, pred, mask):
        bce_loss = self.bce_loss(pred, mask)
        
        dl_loss = self.dice_loss(pred, mask)
        
        return bce_loss + dl_loss * self.lam
        
        
        
        
        
        
        
        
        
    