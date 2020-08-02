import torch
import kornia
import time
import copy
import shutil
import numpy as np
import torch.nn as nn
from graphviz import Digraph
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn.functional as F
from unet.loss import dice_loss
import torch.optim as optim
from data.dataset import *
from unet.pytorch_DPCN import FFT2, UNet, LogPolar, PhaseCorr, Corr2Softmax
from data.dataset_DPCN import *
from tensorboardX import SummaryWriter
from utils.utils import *
def detect_rot_scale(template_rot, source_rot, model_template_rot, model_source_rot, model_corr2softmax_rot, device):
    print("                             ")
    print("                             DETETCTING ROTATION AND SCALE")
    print("                             ")
    template_unet_rot = model_template_rot(template_rot)
    source_unet_rot = model_source_rot(source_rot)

    # for tensorboard visualize
    template_visual_rot = template_unet_rot
    source_visual_rot = source_unet_rot

    # print(np.shape(template_unet_rot))
    # imshow(template_unet_rot)
    # convert to [B,H,W,C]
    template_unet_rot = template_unet_rot.permute(0,2,3,1)
    source_unet_rot = source_unet_rot.permute(0,2,3,1)
    
    template_unet_rot = template_unet_rot.squeeze(-1)
    source_unet_rot = source_unet_rot.squeeze(-1)

    fft_layer = FFT2(device)
    template_fft = fft_layer(template_unet_rot)
    source_fft = fft_layer(source_unet_rot) # [B,H,W,1]

    h = logpolar_filter((source_fft.shape[1],source_fft.shape[2]), device)#highpass((source.shape[1],source.shape[2])) # [H,W]
    template_fft = template_fft.squeeze(-1) * h
    source_fft = source_fft.squeeze(-1) * h
    
    template_fft = template_fft.unsqueeze(-1)
    source_fft = source_fft.unsqueeze(-1)

    # for tensorboard visualize
    template_fft_visual = template_fft.permute(0,3,1,2)
    source_fft_visual = source_fft.permute(0,3,1,2)

    logpolar_layer = LogPolar((template_fft.shape[1], template_fft.shape[2]), device)
    template_logpolar, logbase_rot = logpolar_layer(template_fft)
    source_logpolar, logbase_rot = logpolar_layer(source_fft)

    # for tensorboard visualize
    template_logpolar_visual = template_logpolar.permute(0,3,1,2)
    source_logpolar_visual = source_logpolar.permute(0,3,1,2)

    template_logpolar = template_logpolar.squeeze(-1)
    source_logpolar = source_logpolar.squeeze(-1)
    phase_corr_layer_rs = PhaseCorr(device, logbase_rot, model_corr2softmax_rot)
    rotation_cal, scale_cal, softmax_result_rot, corr_result_rot = phase_corr_layer_rs(template_logpolar, source_logpolar)



# use phasecorr result

  
    print("rotation =", rotation_cal)
    print("scale =", scale_cal)
    # print("gt_angle ", gt_angle)

# # flatten the tensor:
    # b_loss,h_loss,w_loss = groundTruth.shape
    # groundTruth = groundTruth.reshape(b_loss,h_loss*w_loss)
    # softmax_final = softmax_final.reshape(b_loss,h_loss*w_loss)


# set the loss function:
    # compute_loss = torch.nn.KLDivLoss(reduction="sum").to(device)
    # compute_loss = torch.nn.BCEWithLogitsLoss(reduction="sum").to(device)
    compute_loss_rot = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    # compute_loss = torch.nn.MSELoss()
    # compute_loss=torch.nn.L1Loss()

    return rotation_cal, scale_cal

def detect_translation(template_trans, source_trans, rotation, scale, model_template_trans, model_source_trans, model_corr2softmax_trans, device ):
    print("                             ")
    print("                             DETECTING TRANSLATION")
    print("                             ")


# for AGDatase
    b, c, h, w = source_trans.shape
    center = torch.ones(b,2).to(device)
    center[:, 0] = h // 2
    center[:, 1] = w // 2
    angle_rot = torch.ones(b).to(device) * (-rotation.to(device))
    scale_rot = torch.ones(b).to(device) * (1/scale.to(device))
    rot_mat = kornia.get_rotation_matrix2d(center, angle_rot, scale_rot)
    source_trans = kornia.warp_affine(source_trans.to(device), rot_mat, dsize=(h, w))
    # imshow(template_trans[0,:,:])
    # time.sleep(2)
    # imshow(source_trans[0,:,:])
    # time.sleep(2)

    # imshow(template,"temp")
    # imshow(source, "src")
    
    template_unet_trans = model_template_trans(template_trans)
    source_unet_trans = model_source_trans(source_trans)

    # for tensorboard visualize
    template_visual_trans = template_unet_trans
    source_visual_trans = source_unet_trans

    template_unet_trans = template_unet_trans.permute(0,2,3,1)
    source_unet_trans = source_unet_trans.permute(0,2,3,1)

    template_unet_trans = template_unet_trans.squeeze(-1)
    source_unet_trans = source_unet_trans.squeeze(-1)

    (b, h, w) = template_unet_trans.shape
    logbase_trans = torch.tensor(1.)
    phase_corr_layer_xy = PhaseCorr(device, logbase_trans, model_corr2softmax_trans)
    t0, t1, softmax_result_trans, corr_result_trans = phase_corr_layer_xy(template_unet_trans.to(device), source_unet_trans.to(device))

# use phasecorr result

    corr_final_trans = corr_result_trans.clone()
    # corr_visual = corr_final_trans.unsqueeze(-1)
    # corr_visual = corr_visual.permute(0,3,1,2)
    corr_y = torch.sum(corr_final_trans.clone(), 2, keepdim=False)
    # corr_2d = corr_final_trans.clone().reshape(b, h*w)
    # corr_2d = model_corr2softmax(corr_2d)
    corr_y = model_corr2softmax_trans(corr_y)
    input_c = nn.functional.softmax(corr_y.clone(), dim=-1)
    indices_c = np.linspace(0, 1, 256)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, 256))).to(device)
    transformation_y = torch.sum((256 - 1) * input_c * indices_c, dim=-1)
    # transformation_y = torch.argmax(corr_y, dim=-1)

    corr_x = torch.sum(corr_final_trans.clone(), 1, keepdim=False)
    # corr_final_trans = corr_final_trans.reshape(b, h*w)
    corr_x = model_corr2softmax_trans(corr_x)
    input_r = nn.functional.softmax(corr_x.clone(), dim=-1)
    indices_r = np.linspace(0, 1, 256)
    indices_r = torch.tensor(np.reshape(indices_r, (-1, 256))).to(device)
    # transformation_x = torch.argmax(corr_x, dim=-1)
    transformation_x = torch.sum((256 - 1) * input_r * indices_r, dim=-1)

    print("trans x", transformation_x)
    print("trans y", transformation_y)

    trans_mat_affine = torch.Tensor([[[1.0,0.0,transformation_x-128.0],[0.0,1.0,transformation_y-128.0]]]).to(device)
    template_trans = kornia.warp_affine(template_trans.to(device), trans_mat_affine, dsize=(h, w))
    image_aligned = align_image(template_trans[0,:,:], source_trans[0,:,:])
    # imshow(template_trans[0,:,:])
    # time.sleep(2)
    # imshow(source_trans[0,:,:])
    # time.sleep(2)

    return transformation_y, transformation_x, image_aligned, source_trans



