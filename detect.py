from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import copy
from unet.pytorch_DPCN import FFT2, UNet, LogPolar, PhaseCorr, Corr2Softmax
from data.dataset_DPCN import *
import numpy as np
import shutil
from utils.utils import *
import kornia
from data.dataset import *
from utils.detect_utils import *


def detect_model(template_path, source_path, model_template, model_source, model_corr2softmax,\
             model_trans_template, model_trans_source, model_trans_corr2softmax):
    batch_size_inner = 1

    since = time.time()

    # Each epoch has a training and validation phase
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    phase = "val"

    model_template.eval()   # Set model to evaluate mode
    model_source.eval()
    model_corr2softmax.eval()
    model_trans_template.eval()
    model_trans_source.eval()
    model_trans_corr2softmax.eval()
    with torch.no_grad():

        iters = 0
        acc = 0.
        template, _, _, _, _, _ = default_loader(template_path, 256)
        source, _, _, _, _, _ = default_loader(source_path, 256)
        template = template.to(device)
        source = source.to(device)
        template = template.unsqueeze(0)
        template = template.permute(1,0,2,3)
        source = source.unsqueeze(0)
        source = source.permute(1,0,2,3)

        iters += 1    
        since = time.time()
        rotation_cal, scale_cal = detect_rot_scale(template, source,\
                                     model_template, model_source, model_corr2softmax, device )
        tranformation_y, tranformation_x, image_aligned, source_rotated = detect_translation(template, source, rotation_cal, scale_cal, \
                                            model_trans_template, model_trans_source, model_trans_corr2softmax, device)
        time_elapsed = time.time() - since
        # print('in detection time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("in detection time", time_elapsed)
        plot_and_save_result(template[0,:,:], source[0,:,:], source_rotated[0,:,:], image_aligned)





checkpoint_path = "./checkpoints/laser_sat_qsdjt_9epoch.pt"
template_path = "./demo/temp_2.jpg"
source_path = "./demo/src_2.jpg"

load_pretrained =True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The devices that the code is running on:", device)
# device = torch.device("cpu")
batch_size = 1
num_class = 1
start_epoch = 0
model_template = UNet(num_class).to(device)
model_source = UNet(num_class).to(device)
model_corr2softmax = Corr2Softmax(200., 0.).to(device)
model_trans_template = UNet(num_class).to(device)
model_trans_source = UNet(num_class).to(device)
model_trans_corr2softmax = Corr2Softmax(200., 0.).to(device)

optimizer_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
optimizer_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
optimizer_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)
optimizer_trans_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
optimizer_trans_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
optimizer_trans_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)

if load_pretrained:
    model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
    _, _, _, _, _, _,\
        start_epoch = load_checkpoint(\
                                    checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                    optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)

detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax)
        

                                     





