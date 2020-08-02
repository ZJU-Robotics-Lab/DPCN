import sys
import os
sys.path.append(os.path.abspath("../unet"))
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
from unet.loss import dice_loss
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
import time
import copy
import numpy as np
import shutil
import math
from PIL import Image
import kornia
import cv2


def logpolar_filter(shape, device):
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
    filt = torch.from_numpy(filt).to(device)
    return filt

def roll_n(X, axis, n):
    # print("x")
    # print(X)

    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift = n_shift+1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def softargmax(input, device, beta=100):
    *_, h, w = input.shape

    # input = input.reshape(*_, h * w)
    input = input.squeeze(0)
    input = input.reshape(1, h * w)
    input = input * 6000
    result = torch.sum(torch.exp(input).to(device) / torch.sum(torch.exp(input).to(device)).to(device) * torch.arange(0,h*w).to(device)).to(device)
    col = result // h
    row = result % col
    
    return result   

def softargmax2d(input, device, beta=10000):
    *_, h, w = input.shape

    input_orig = input.reshape(*_, h * w)
    # print(torch.max(input_orig))
    # print(torch.min(input_orig))
    beta_t = 100000. / torch.max(input_orig).to(device)
    # print(input_orig)
    # print(beta * input_orig)
    input_d = nn.functional.softmax(beta_t * input_orig, dim=-1)
    input_orig.retain_grad()
    # print(torch.argmax(input))
    # print(torch.max(input_d))
    # print(torch.min(input_d))
    # print(torch.sum(input))

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).to(device)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).to(device)

    result_r = torch.sum((h - 1) * input_d * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input_d * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)
    # result.sum().backward(retain_graph=True)
    # print(input_orig.grad)

    return result 

def softmax2d(input, device, beta=10000):
    *_, h, w = input.shape
    
    input_orig = input.reshape(*_, h * w)
    beta_t = 100. / torch.max(input_orig).to(device)
    input_d = nn.functional.softmax(1000 * input_orig, dim=-1)
    soft_r = input_d.reshape(*_,h,w)
    # soft_r.retain_grad()
    # print("softmax grad =======", soft_r.grad)
    return soft_r

def GT_angle_convert(this_gt,size):
    for batch_num in range(this_gt.shape[0]):
        if this_gt[batch_num] > 90:
            this_gt[batch_num] = this_gt[batch_num].clone() - 90
        else:
            this_gt[batch_num] = this_gt[batch_num].clone() + 90
        this_gt[batch_num] = this_gt[batch_num].clone()*size/180
        this_gt[batch_num] = this_gt[batch_num].clone()//1 + (this_gt[batch_num].clone()%1+0.5)//1
        if this_gt[batch_num].long() == size:
            this_gt[batch_num] = this_gt[batch_num] - 1
    return this_gt.long()
def GT_scale_convert(scale_gt,logbase,size):
    for batch_num in range(scale_gt.shape[0]):
            scale_gt[batch_num] = torch.log10(1/scale_gt[batch_num].clone())/torch.log10(logbase)+128.
    return scale_gt.long()

def GT_trans_convert(this_trans, size):
    this_trans = (this_trans.clone() + size[0] // 2)
    # gt_converted = this_trans[:,1] * size[0] + this_trans[:,0]
    # gt_converted = this_trans[:,0]
    gt_converted = this_trans
# # create a gt for kldivloss
    # kldiv_gt = torch.zeros(this_trans.clone().shape[0],size[0],size[1])
    # gauss_blur = kornia.filters.GaussianBlur2d((5, 5), (5, 5))
    # for batch_num in range(this_trans.clone().shape[0]):
    #     kldiv_gt[batch_num, this_trans.clone()[batch_num,0].long(), this_trans.clone()[batch_num,1].long()] = 1

    # kldiv_gt = torch.unsqueeze(kldiv_gt.clone(), dim = 0)
    # kldiv_gt = kldiv_gt.permute(1,0,2,3)
    # kldiv_gt = gauss_blur(kldiv_gt.clone())
    # kldiv_gt = kldiv_gt.permute(1,0,2,3)
    # kldiv_gt = torch.squeeze(kldiv_gt.clone(), dim = 0)
    # (b, h, w) = kldiv_gt.shape
    # kldiv_gt = kldiv_gt.clone().reshape(b, h*w)
# # Create GT for Pooling data
#     gt_pooling = torch.floor(this_trans.clone()/4)
    return gt_converted.long()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] = metrics['bce']+bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] = metrics['dice']+dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] = metrics['loss']+loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
def imshow(tensor, title=None):
    image = tensor.cpu().detach().numpy()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    plt.imshow(image, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
def heatmap_imshow(tensor, title=None):
    image = tensor.cpu().detach().numpy()  # we clone the tensor to not do changes on it
    image = gaussian_filter(image, sigma = 5, mode = 'nearest')
    plt.imshow(image, cmap="jet", interpolation="hamming")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
def align_image(template, source):
    template = template.cpu().detach().numpy()
    source = source.cpu().detach().numpy()
    template = template.squeeze(0)  # remove the fake batch dimension
    source = source.squeeze(0)  # remove the fake batch dimension
    dst = cv2.addWeighted(template, 0.4, source, 0.6, 0)
    # plt.imshow(dst, cmap="gray")
    # plt.show()  # pause a bit so that plots are updated
    return dst

def plot_and_save_result(template, source, rotated, dst):
    template = template.cpu().detach().numpy()
    source = source.cpu().detach().numpy()
    template = template.squeeze(0)  # remove the fake batch dimension
    source = source.squeeze(0)  # remove the fake batch dimension
    rotated = rotated.cpu().detach().numpy()
    rotated = rotated.squeeze(0)


    result = plt.figure()
    result_t = result.add_subplot(1,4,1)
    result_t.set_title("Template")
    result_t.imshow(template, cmap="gray").axes.get_xaxis().set_visible(False)
    result_t.imshow(template, cmap="gray").axes.get_yaxis().set_visible(False)

    result_s = result.add_subplot(1,4,2)
    result_s.set_title("Source")
    result_s.imshow(source, cmap="gray").axes.get_xaxis().set_visible(False)
    result_s.imshow(source, cmap="gray").axes.get_yaxis().set_visible(False)

    result_r = result.add_subplot(1,4,3)
    result_r.set_title("Rotated Source")
    result_r.imshow(rotated, cmap="gray").axes.get_xaxis().set_visible(False)
    result_r.imshow(rotated, cmap="gray").axes.get_yaxis().set_visible(False)

    result_d = result.add_subplot(1,4,4)
    result_d.set_title("Destination")
    result_d.imshow(dst, cmap="gray").axes.get_xaxis().set_visible(False)
    result_d.imshow(dst, cmap="gray").axes.get_yaxis().set_visible(False)
    plt.savefig("Result.png")
    plt.show()
   


def save_checkpoint(state, is_best, checkpoint_dir):
    file_path = checkpoint_dir + 'checkpoint.pt'
    torch.save(state, file_path)
    if is_best:
        best_fpath = checkpoint_dir + 'best_model.pt'
        shutil.copyfile(file_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model_template, model_source, model_c2s, model_trans_template, model_trans_source, model_trans_c2s,\
                                    optimizer_temp, optimizer_src, optimizer_c2s, optimizer_trans_temp, optimizer_trans_src, optimizer_trans_c2s, device):

    if (device == torch.device('cpu')):
        print("using cpu")
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_fpath, map_location=device)


    model_template.load_state_dict(checkpoint['state_dict_temp'])
    model_source.load_state_dict(checkpoint['state_dict_src'])
    model_c2s.load_state_dict(checkpoint['state_dict_c2s'])
    optimizer_temp.load_state_dict(checkpoint['optimizer_temp'])
    optimizer_src.load_state_dict(checkpoint['optimizer_src'])
    optimizer_c2s.load_state_dict(checkpoint['optimizer_c2s'])

    model_trans_template.load_state_dict(checkpoint['state_dict_trans_temp'])
    model_trans_source.load_state_dict(checkpoint['state_dict_trans_src'])
    model_trans_c2s.load_state_dict(checkpoint['state_dict_trans_c2s'])
    optimizer_trans_temp.load_state_dict(checkpoint['optimizer_trans_temp'])
    optimizer_trans_src.load_state_dict(checkpoint['optimizer_trans_src'])
    optimizer_trans_c2s.load_state_dict(checkpoint['optimizer_trans_c2s'])

    return model_template, model_source, model_c2s, model_trans_template, model_trans_source, model_trans_c2s,\
                    optimizer_temp, optimizer_src, optimizer_c2s, optimizer_trans_temp, optimizer_trans_src, optimizer_trans_c2s, \
                    checkpoint['epoch']

def load_trans_checkpoint(checkpoint_fpath, model_trans_template, model_trans_source,\
                                     device):

    if (device == torch.device('cpu')):
        print("using cpu")
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_fpath)

    model_trans_template.load_state_dict(checkpoint['state_dict_t_src'])
    model_trans_source.load_state_dict(checkpoint['state_dict_t_temp'])

    return  model_trans_template, model_trans_source,\
                    checkpoint['epoch']

def load_rot_checkpoint(checkpoint_rpath, model_template, model_source, model_c2s,\
                                    optimizer_temp, optimizer_src, optimizer_c2s, device):

    if (device == torch.device('cpu')):
        print("using cpu")
        checkpoint = torch.load(checkpoint_rpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_rpath)
    model_template.load_state_dict(checkpoint['state_dict_temp'])
    model_source.load_state_dict(checkpoint['state_dict_src'])
    model_c2s.load_state_dict(checkpoint['state_dict_c2s'])
    optimizer_temp.load_state_dict(checkpoint['optimizer_temp'])
    optimizer_src.load_state_dict(checkpoint['optimizer_src'])
    optimizer_c2s.load_state_dict(checkpoint['optimizer_c2s'])

    return model_template, model_source, model_c2s, optimizer_temp, optimizer_src, optimizer_c2s