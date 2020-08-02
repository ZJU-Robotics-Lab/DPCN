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
from utils.validate_utils import *
import argparse


def val_model(model_template, model_source, model_corr2softmax,\
             model_trans_template, model_trans_source, model_trans_corr2softmax, \
             writer_val, iters, dsnt, dataloader, batch_size_val, device, epoch):

    # for the use of visualizing the validation properly on the tensorboard
    iters -= 500
    phase = "val"
    loss_list = []
    rot_list = []
    model_template.eval()   # Set model to evaluate mode
    model_source.eval()
    model_corr2softmax.eval()
    model_trans_template.eval()
    model_trans_source.eval()
    model_trans_corr2softmax.eval()
    acc_x = np.zeros(20)
    acc_y = np.zeros(20)
    acc = 0.

    with torch.no_grad():

        for template, source, groundTruth_number, gt_scale,  gt_trans in dataloader(batch_size_val)[phase]:
            template = template.to(device)
            source = source.to(device)
            iters += 1    
            # imshow(template[0,:,:])
            # plt.show()
            # imshow(source[0,:,:])
            # plt.show()
            # print("gtSCALE~~~~",gt_scale)
            loss_rot, loss_scale, scale_cal, loss_l1_rot, loss_mse_rot, loss_l1_scale, loss_mse_scale \
                    = validate_rot_scale(template.clone(), source.clone(), groundTruth_number.clone(), gt_scale.clone(),\
                                         model_template, model_source, model_corr2softmax, device )
            loss_y, loss_x, total_loss, loss_l1_x,loss_l1_y,loss_mse_x, loss_mse_y \
                    = validate_translation(template.clone(), source.clone(), groundTruth_number.clone(), gt_scale.clone(), gt_trans.clone(), \
                                            model_trans_template, model_trans_source, model_trans_corr2softmax,acc_x, acc_y, dsnt, device)


            # loss = compute_loss(corr_final, gt_angle)
            total_rs_loss = loss_rot + loss_scale
            loss_list.append(total_rs_loss.tolist())
            writer_val.add_scalar('LOSS ROTATION', loss_rot.detach().cpu().numpy(), iters)
            writer_val.add_scalar('LOSS SCALE', loss_scale.detach().cpu().numpy(), iters)
            writer_val.add_scalar('LOSS X', loss_x.detach().cpu().numpy(), iters)
            writer_val.add_scalar('LOSS Y', loss_y.detach().cpu().numpy(), iters)

            writer_val.add_scalar('LOSS ROTATION L1', loss_l1_rot.item(), iters)
            writer_val.add_scalar('LOSS ROTATION MSE', loss_mse_rot.item(), iters)
            writer_val.add_scalar('LOSS SCALE L1', loss_l1_scale.item(), iters)
            writer_val.add_scalar('LOSS SCALE MSE', loss_mse_scale.item(), iters)

            writer_val.add_scalar('LOSS X L1', loss_l1_x.item(), iters)
            writer_val.add_scalar('LOSS X MSE', loss_mse_x.item(), iters)
            writer_val.add_scalar('LOSS Y L1', loss_l1_y.item(), iters)
            writer_val.add_scalar('LOSS Y MSE', loss_mse_y.item(), iters)

    X = np.linspace(0, 19, 20)
    fig = plt.figure()
    plt.bar(X,acc_x/1000)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")    
    
    plt.savefig("./checkpoints/barChart/x/"+ str(epoch) + "_toy_barChartX_top1.jpg")

    Y = np.linspace(0, 19, 20)
    fig = plt.figure()
    plt.bar(Y,acc_y/1000)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")    
    
    plt.savefig("./checkpoints/barChart/y/"+ str(epoch) + "_toy_barChartY_top1.jpg")
    return loss_list


# Passing a bunch of parameters
parser_val = argparse.ArgumentParser(description="DPCN Network Validation")
parser_val.add_argument('--only_valid', action='store_true', default=False)
parser_val.add_argument('--cpu', action='store_true', default=False)
parser_val.add_argument('--load_path', type=str, default="./checkpoints/checkpoint.pt")
parser_val.add_argument('--simulation', action='store_true', default=False)
parser_val.add_argument('--use_dsnt', action='store_true', default=False)
parser_val.add_argument('--batch_size_val', type=int, default=2)
parser_val.add_argument('--val_writer_path', type=str, default="./checkpoints/log/val/")
args_val = parser_val.parse_args()

if args_val.only_valid:
    epoch = 1
    checkpoint_path = args_val.load_path
    device = torch.device("cuda:0" if not args_val.cpu else "cpu")
    print("The devices that the code is running on:", device)
    writer_val = SummaryWriter(log_dir=args_val.val_writer_path)
    batch_size_val = args_val.batch_size_val
    dataloader = generate_dataloader if args_val.simulation else DPCNdataloader
    dsnt = args_val.use_dsnt


    num_class = 1
    start_epoch = 0
    iters = 0


# create a shell model for checkpoint loader to load into
    model_template = UNet(num_class).to(device)
    model_source = UNet(num_class).to(device)
    model_corr2softmax = Corr2Softmax(200., 0.).to(device)
    model_trans_template = UNet(num_class).to(device)
    model_trans_source = UNet(num_class).to(device)
    model_trans_corr2softmax = Corr2Softmax(11.72, 0.).to(device)

    optimizer_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
    optimizer_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
    optimizer_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)
    optimizer_trans_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
    optimizer_trans_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
    optimizer_trans_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)


# load checkpoint
    model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
    optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s,\
        start_epoch = load_checkpoint(\
                                    checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                    optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)

# Entering the mean loop of Validation
    loss_list = val_model(model_template, model_source, model_corr2softmax, \
        model_trans_template, model_trans_source, model_trans_corr2softmax, \
        writer_val, iters, dsnt, dataloader, batch_size_val, device, epoch)
            

                                     





