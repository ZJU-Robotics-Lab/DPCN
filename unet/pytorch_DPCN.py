import sys
import os
sys.path.append(os.path.abspath(".."))
import torch
import torch.nn as nn
import numpy as np
from phase_correlation.phase_corr import phase_corr
from log_polar.log_polar import polar_transformer
from utils.utils import *
print("sys path", sys.path)
from utils.utils import *

class LogPolar(nn.Module):
    def __init__(self, out_size, device):
        super(LogPolar, self).__init__()
        self.out_size = out_size
        self.device = device

    def forward(self, input):
        return polar_transformer(input, self.out_size, self.device) 


class PhaseCorr(nn.Module):
    def __init__(self, device, logbase, modelc2s, trans=False):
        super(PhaseCorr, self).__init__()
        self.device = device
        self.logbase = logbase
        self.trans = trans
        self.modelc2s = modelc2s

    def forward(self, template, source):
        return phase_corr(template, source, self.device, self.logbase, self.modelc2s, trans=self.trans)

class FFT2(nn.Module):
    def __init__(self, device):
        super(FFT2, self).__init__()
        self.device = device

    def forward(self, input):
        median_output = torch.rfft(input, 2, onesided=False)
        median_output_r = median_output[:, :, :, 0]
        median_output_i = median_output[:, :, :, 1]
        # print("median_output r", median_output_r)
        # print("median_output i", median_output_i)
        output = torch.sqrt(median_output_r ** 2 + median_output_i ** 2 + 1e-15)
        # output = median_outputW_r
        output = fftshift2d(output)
        # h = logpolar_filter((output.shape[1],output.shape[2]), self.device)
        # output = output.squeeze(0) * h
        # output = output.unsqueeze(-1)
        output = output.unsqueeze(-1)
        return output

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)

        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)   
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class Corr2Softmax(nn.Module):

    def __init__(self, weight, bias):

        super(Corr2Softmax, self).__init__()
        softmax_w = torch.tensor((weight), requires_grad=True)
        softmax_b = torch.tensor((bias), requires_grad=True)
        self.softmax_w = torch.nn.Parameter(softmax_w)
        self.softmax_b = torch.nn.Parameter(softmax_b)
        self.register_parameter("softmax_w",self.softmax_w)
        self.register_parameter("softmax_b",self.softmax_b)
    def forward(self, x):
        x1 = self.softmax_w*x + self.softmax_b
        # print("w = ",self.softmax_w, "b = ",self.softmax_b)
        # x1 = 1000. * x
        return x1




