import sys
import os
sys.path.append(os.path.abspath(".."))

import cv2
import math
import torch
import kornia
import numpy as np
import torch.nn as nn
from numpy.fft import fft2, ifft2, fftshift
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from utils.utils import *
from log_polar.log_polar import *

def phase_corr(a, b, device, logbase, modelc2s, trans=False):
# a: template; b: source
# imshow(a.squeeze(0).float())
    G_a = torch.rfft(a, 2, onesided=False)
    G_b = torch.rfft(b, 2, onesided=False)
    eps=1e-15

    real_a = G_a[:, :, :, 0]
    real_b = G_b[:, :, :, 0]
    imag_a = G_a[:, :, :, 1]
    imag_b = G_b[:, :, :, 1]

# compute a * b.conjugate; shape=[B,H,W,C]
    R = torch.FloatTensor(G_a.shape[0], G_a.shape[1], G_a.shape[2], 2).to(device)
    R[:, :, :, 0] = real_a * real_b + imag_a * imag_b
    R[:, :, :, 1] = real_a * imag_b - real_b * imag_a

    r0 = torch.sqrt(real_a ** 2 + imag_a ** 2 + eps) * torch.sqrt(real_b ** 2 + imag_b ** 2 + eps)
    R[:, :, :, 0] = R[:, :, :, 0].clone()/(r0 + eps).to(device)
    R[:, :, :, 1] = R[:, :, :, 1].clone()/(r0 + eps).to(device)

    r = torch.ifft(R, 2)
    r_real = r[:, :, :, 0]
    r_imag = r[:, :, :, 1]
    r = torch.sqrt(r_real ** 2 + r_imag ** 2 + eps)
    r = fftshift2d(r)
    if trans:
        r[:,0:60,:]=0.
        r[:,G_a.shape[1]-60:G_a.shape[1], :] = 0.
        r[:,:, 0:60]=0.
        r[:, :, G_a.shape[1]-60:G_a.shape[1]] = 0.
    # imshow(r[0,:,:])
    # plt.show()
# feed the result of phase correlation to the NET
    softargmax_input = modelc2s(r.clone())
# suppress the output to angle and scale
    angle_resize_out_tensor = torch.sum(softargmax_input.clone(), 2, keepdim=False)
    scale_reszie_out_tensor = torch.sum(softargmax_input.clone(), 1, keepdim=False)
# get the argmax of the angle and the scale
    angle_out_tensor = torch.argmax(angle_resize_out_tensor.clone().detach(), dim=-1)
    scale_out_tensor = torch.argmax(scale_reszie_out_tensor.clone().detach(), dim=-1)

#calculate angle
    angle = angle_out_tensor*180.00/r.shape[1]
    for batch_num in range(angle.shape[0]):
        if angle[batch_num].item() > 90:
            angle[batch_num] -= 90.00
        else:
            angle[batch_num] += 90.00
# compute the softmax in case any needs
    softmax_result = softmax2d(softargmax_input, device)
    # imshow(softmax_result[0,:,:])
    # plt.show()
# calculate scale
    logbase = logbase.to(device)

    sca_f = scale_out_tensor.clone()-r.shape[2] // 2
    scale = 1 / torch.pow(logbase, sca_f.float())#logbase ** sca_f

    return [angle, scale, softmax_result,r]
    
def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    i1 = torch.cos(torch.linspace(-np.pi/2.0, np.pi/2.0, shape[0]))
    i2 = torch.cos(torch.linspace(-np.pi/2.0, np.pi/2.0, shape[1]))
    x = torch.einsum('i,j->ij', i1, i2)
    return (1.0 - x) * (1.0 - x)

def logpolar_filter(shape):
    """
    Make a radial cosine filter for the logpolar transform.
    This filter suppresses low frequencies and completely removes
    the zero freq.
    """
    yy = np.linspace(- np.pi / 2., np.pi / 2., shape[0])[:, np.newaxis]
    xx = np.linspace(- np.pi / 2., np.pi / 2., shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy ** 2 + xx ** 2)
    filt = 1.0 - np.cos(rads) ** 2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1
    filt = torch.from_numpy(filt)
    return filt

class LogPolar(nn.Module):
    def __init__(self, out_size, device):
        super(LogPolar, self).__init__()
        self.out_size = out_size
        self.device = device

    def forward(self, input):
        return polar_transformer(input, self.out_size, self.device) 


class PhaseCorr(nn.Module):
    def __init__(self, device, logbase, trans=False):
        super(PhaseCorr, self).__init__()
        self.device = device
        self.logbase = logbase
        self.trans = trans

    def forward(self, template, source):
        return phase_corr(template, source, self.device, self.logbase, trans=self.trans)

##############################################
# grad check
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# x = torch.randn((1, 4, 3, 1), requires_grad=True)
# y = torch.ones((1, 4, 3, 1),requires_grad=True)

# logpolar_layer = LogPolar((x.shape[1], x.shape[2]), device)
# x_logpolar, log_base = logpolar_layer(x)
# y_logpolar, log_base = logpolar_layer(y)

# x_logpolar = x_logpolar.squeeze(-1)
# y_logpolar = y_logpolar.squeeze(-1)
# y_logpolar.retain_grad()

# phase_corr_layer_rs = PhaseCorr(device, log_base)
# angle, scale, _ = phase_corr_layer_rs(x_logpolar, y_logpolar)

# rx = torch.ifft(x, 2)
# r_real = rx[:, :, :, 0]
# r_real.retain_grad()
# r_imag = rx[:, :, :, 1]
# r = torch.sqrt(r_real ** 2 + r_imag ** 2)
# r = fftshift2d(r)
# r.sum().backward()
# print(r_real.grad)


# loss = nn.L1Loss()
# loss = loss(x_logpolar)
# print(x.device)
# x_logpolar.sum().backward()
# angle.backward()
# print(y_logpolar.grad)

# ###############################################
# # overall check

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# trans = transforms.Compose([
#         transforms.ToTensor(),
#     ])

# # read images
# template = cv2.imread("./1_-70.jpg", 0)
# source = cv2.imread("./7_33.jpg", 0)
# t_x = 0.
# t_y = 0.
# # gt rot and scale
# col, row = template.shape
# center = (col // 2, row // 2)

# N = np.float32([[1,0,t_x],[0,1,t_y]])
# source = cv2.warpAffine(source, N, (col, row))
# # M = cv2.getRotationMatrix2D(center, 0., 1.)
# # source = cv2.warpAffine(source, M, (col, row))


# template = trans(template)
# source = trans(source)
# # template = template.unsqueeze(0)
# # source = source.unsqueeze(0)
# # imshow(template)
# # plt.show()
# # imshow(source)
# # plt.show()

# # [B,H,W,C]
# template = template.unsqueeze(0)
# template = template.permute(0,2,3,1)
# source = source.unsqueeze(0)
# source = source.permute(0,2,3,1)
# template_img = template.squeeze(-1)
# source_img = source.squeeze(-1)

# # fft
# template = torch.rfft(template_img, 2, onesided=False)
# source = torch.rfft(source_img, 2, onesided=False)

# # fftshift
# template_r = template[:, :, :, 0]
# template_i = template[:, :, :, 1]
# template = torch.sqrt(template_r ** 2 + template_i ** 2)

# source_r = source[:, :, :, 0]
# source_i = source[:, :, :, 1]
# source = torch.sqrt(source_r ** 2 + source_i ** 2)

# template = fftshift2d(template)
# source = fftshift2d(source) # [B,H,W]

# # highpass filter
# h = logpolar_filter((source.shape[1],source.shape[2]))#highpass((source.shape[1],source.shape[2])) # [H,W]
# template = template.squeeze(0) * h
# source = source.squeeze(0) * h

# # print(template)
# # imshow(template.squeeze(0))
# # change size
# template = template.unsqueeze(-1)
# source = source.unsqueeze(-1)
# template = template.unsqueeze(0)
# source = source.unsqueeze(0)

# # log_polar
# template, logbase = polar_transformer(template, (source.shape[1], source.shape[2]), device)
# source, logbase = polar_transformer(source, (source.shape[1], source.shape[2]), device)

# source = source.squeeze(-1)
# template = template.squeeze(-1)
# # imshow(template.squeeze(0))

# # phase corr
# rot, scale, _,_,_ = phase_corr(template, source, device, logbase, trans=False)

# # angle = -rot * math.pi/180
# center = torch.ones(1,2).to(device)
# center[:, 0] = col // 2
# center[:, 1] = row // 2
# rot =  torch.zeros(1).to(device)
# rot_mat = kornia.get_rotation_matrix2d(center, -rot, 1/scale)
# _, h, w = source_img.shape
# new_source_img = kornia.warp_affine(source_img.unsqueeze(1).to(device), rot_mat, dsize=(h, w))

# # theta = torch.tensor([
# #     [math.cos(angle), math.sin(-angle), 0],
# #     [math.sin(angle), math.cos(angle), 0]
# # ], dtype=torch.float)
# # grid = F.affine_grid(theta.unsqueeze(0), source_img.unsqueeze(0).size())
# # output = F.grid_sample(source_img.unsqueeze(0), grid)

# # theta = torch.tensor([
# #     [scale, 0., 0],
# #     [0., scale, 0]
# # ], dtype=torch.float)
# # grid = F.affine_grid(theta.unsqueeze(0), source_img.unsqueeze(0).size())
# # output = F.grid_sample(output[0].unsqueeze(0), grid)
# # new_source_img = output[0].to(device)
# # imshow(source_img)

# # imshow(new_source_img.squeeze(1))
# # imshow(template_img)

# t0, t1, success, _, r = phase_corr(template_img.to(device), new_source_img.squeeze(1), device, logbase, trans=True)
# imshow(r)
# plt.show()
# print("success", success)
# # if success == 1:
# #     rot += 180
# #     print("rot+= 180")

# rot += success.squeeze(0)*180

# if rot < -180.0:
#     rot += 360.0
# elif rot > 180.0:
#     rot -= 360.0

# print(rot)
# print(scale)
# print(t0,t1)