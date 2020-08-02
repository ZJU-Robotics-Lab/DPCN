
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
from utils.train_utils import *
from validate import val_model
import argparse


# adding a bunch of parameters for an easy access
parser = argparse.ArgumentParser(description="DPCN Network Training")

parser.add_argument('--cpu', action='store_true', default=False, help="The Program will use cpu for the training")
parser.add_argument('--save_path', type=str, default="./checkpoints/", help="The path to save the checkpoint of every epoch")
parser.add_argument('--simulation', action='store_true', default=False, help="The training will be applied on a randomly generated simulation dataset")
parser.add_argument('--load_pretrained', action='store_true', default=False, help="Choose whether to use a pretrained model to fine tune")
parser.add_argument('--load_path', type=str, default="./checkpoints/checkpoint.pt", help="The path to load a pretrained checkpoint")
parser.add_argument('--load_optimizer', action='store_true', default=False, help="When using a pretrained model, options of loading it's optimizer")
parser.add_argument('--pretrained_mode', type=str, default="all", help="Three options: 'all' for loading rotation and translation; 'rot' for loading only rotation; 'trans' for loading only translation")
parser.add_argument('--use_dsnt', action='store_true', default=False, help="When enabled, the loss will be calculated via DSNT and MSELoss, or it will use a CELoss")
parser.add_argument('--batch_size_train', type=int, default=2, help="The batch size of training")
parser.add_argument('--batch_size_val', type=int, default=2, help="The batch size of validation")
parser.add_argument('--train_writer_path', type=str, default="./checkpoints/log/train/", help="Where to write the Log of training")
parser.add_argument('--val_writer_path', type=str, default="./checkpoints/log/val/", help="Where to write the Log of validation")
args = parser.parse_args()

writer = SummaryWriter(log_dir=args.train_writer_path)
writer_val = SummaryWriter(log_dir=args.val_writer_path)
np.set_printoptions(threshold=np.inf)


def train_model(model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                 optimizer_temp, optimizer_src, optimizer_c2s, optimizer_trans_temp, optimizer_trans_src, optimizer_trans_c2s,\
                scheduler_temp, scheduler_src, scheduler_trans_temp, scheduler_trans_src,\
                save_path, start_epoch, num_epochs=25):
    best_loss = 1e10
    iters = 0

    for epoch in range(start_epoch , start_epoch + num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                
                for param_group in optimizer_temp.param_groups:
                    print("LR", param_group['lr'])

                model_template.train()  # Set model to training mode
                model_source.train()
                model_corr2softmax.train()
                model_trans_template.train()
                model_trans_source.train()
                model_trans_corr2softmax.train()
            else:
                model_template.eval()   # Set model to evaluate mode
                model_source.eval()
                model_corr2softmax.eval()
                model_trans_template.eval()
                model_trans_source.eval()
                model_trans_corr2softmax.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            if phase == 'train':
                for template, source, groundTruth_number, scale_gt, gt_trans in dataloader(batch_size)[phase]:
                    iters = iters + 1
                    template = template.to(device)
                    source = source.to(device)
                    torch.autograd.set_detect_anomaly(True)

                    # zero the parameter gradients
                    optimizer_temp.zero_grad()
                    optimizer_src.zero_grad()
                    optimizer_c2s.zero_grad()
                    optimizer_trans_temp.zero_grad()
                    optimizer_trans_src.zero_grad()
                    optimizer_trans_c2s.zero_grad()

        # forward
                    loss_rot, loss_scale, loss_l1_rot, loss_mse_rot, loss_l1_scale, loss_mse_scale, template_visual_rot, source_visual_rot \
                            = train_rot_scale(template, source, groundTruth_number.clone(), scale_gt.clone(),\
                                                 model_template, model_source, model_corr2softmax, phase, device )
                    loss_y, loss_x, total_loss, loss_l1_x,loss_l1_y,loss_mse_x, loss_mse_y, template_visual_trans, source_visual_trans \
                            = train_translation(template, source, groundTruth_number.clone(), scale_gt.clone(), gt_trans, \
                                                    model_trans_template, model_trans_source, model_trans_corr2softmax, phase, dsnt, device)


        # backward + optimize only if in training phase:
                    if phase == 'train':
                        # print(iters)
                        with torch.autograd.detect_anomaly():
                            total_loss.backward(retain_graph=False)
                            loss_rot.backward(retain_graph=True)
                            # loss_l1_rot.backward(retain_graph=False)
                            # loss_scale.backward(retain_graph=True)
                            # loss_x.backward(retain_graph=True)
                            # loss_y.backward(retain_graph=True)
                        optimizer_temp.step()
                        optimizer_src.step()
                        optimizer_c2s.step()
                        optimizer_trans_temp.step()
                        optimizer_trans_src.step()
                        optimizer_trans_c2s.step()
                        writer.add_scalar('LOSS ROTATION', loss_rot.detach().cpu().numpy(), iters)
                        writer.add_scalar('LOSS SCALE', loss_scale.detach().cpu().numpy(), iters)
                        writer.add_scalar('LOSS X', loss_x.detach().cpu().numpy(), iters)
                        writer.add_scalar('LOSS Y', loss_y.detach().cpu().numpy(), iters)

                        writer.add_scalar('LOSS ROTATION L1', loss_l1_rot.item(), iters)
                        writer.add_scalar('LOSS ROTATION MSE', loss_mse_rot.item(), iters)
                        writer.add_scalar('LOSS SCALE L1', loss_l1_scale.item(), iters)
                        writer.add_scalar('LOSS SCALE MSE', loss_mse_scale.item(), iters)

                        writer.add_scalar('LOSS X L1', loss_l1_x.item(), iters)
                        writer.add_scalar('LOSS X MSE', loss_mse_x.item(), iters)
                        writer.add_scalar('LOSS Y L1', loss_l1_y.item(), iters)
                        writer.add_scalar('LOSS Y MSE', loss_mse_y.item(), iters)
                       
                        writer.add_image("temp_input", template[0,:,:].cpu(), iters)
                        writer.add_image("src_input", source[0,:,:].cpu(), iters)
                        writer.add_image("unet_temp_rot", template_visual_rot[0,:,:].cpu(), iters)
                        writer.add_image("unet_src_rot", source_visual_rot[0,:,:].cpu(), iters)
                        writer.add_image("unet_temp_trans", template_visual_trans[0,:,:].cpu(), iters)
                        writer.add_image("unet_src_trans", source_visual_trans[0,:,:].cpu(), iters)
                        # writer.add_image("fft_temp", template_fft_visual[0,:,:].detach().cpu(), iters)
                        # writer.add_image("fft_src", source_fft_visual[0,:,:].detach().cpu(), iters)
                        # writer.add_image("logpolar_temp", template_logpolar_visual[0,:,:].cpu(), iters)
                        # writer.add_image("logpolar_src", source_logpolar_visual[0,:,:].cpu(), iters)
                        # writer.add_image("new", new_source_img[0,:,:].cpu())
                        
                # statistics
                epoch_samples = epoch_samples + template.size(0)


            checkpoint = {'epoch': epoch + 1,
                      'state_dict_temp': model_template.state_dict(),
                      'optimizer_temp': optimizer_temp.state_dict(),
                      'state_dict_src': model_source.state_dict(),
                      'optimizer_src': optimizer_src.state_dict(),
                      'state_dict_c2s': model_corr2softmax.state_dict(),
                      'optimizer_c2s': optimizer_c2s.state_dict(),
                      'state_dict_trans_temp': model_trans_template.state_dict(),
                      'optimizer_trans_temp': optimizer_trans_temp.state_dict(),
                      'state_dict_trans_src': model_trans_source.state_dict(),
                      'optimizer_trans_src': optimizer_trans_src.state_dict(),
                      'state_dict_trans_c2s': model_trans_corr2softmax.state_dict(),
                      'optimizer_trans_c2s': optimizer_trans_c2s.state_dict()}

            if phase == 'val':
                print("in val")
                loss_list = val_model(model_template, model_source, model_corr2softmax,\
                                                model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                                writer_val, iters, dsnt, dataloader, batch_size_val, device, epoch)
                epoch_loss = np.mean(loss_list)
                print("epoch_loss", epoch_loss)
                print("best_loss", best_loss)
                # print("accuracy = ", acc)
                if epoch_loss < best_loss:
                    is_best = True
                    best_loss = epoch_loss
                else:
                    is_best = False
                save_checkpoint(checkpoint, is_best, save_path)

               
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


        scheduler_temp.step()
        scheduler_src.step()
        scheduler_trans_temp.step()
        scheduler_trans_src.step()
    
    print('Best val loss: {:4f}'.format(best_loss))

    return model_template, model_source



save_path = args.save_path
checkpoint_path = args.load_path
load_pretrained = args.load_pretrained
load_optimizer = args.load_optimizer
simulation = args.simulation
dsnt = args.use_dsnt
load_pretrained_mode = args.pretrained_mode
batch_size = args.batch_size_train
batch_size_val = args.batch_size_val
dataloader = generate_dataloader if simulation else DPCNdataloader
device = torch.device("cuda:0" if not args.cpu else "cpu")
print("The devices that the code is running on:", device)
print("batch size is ",batch_size)


# to create models for rotations and translations for source images and template images
num_class = 1
start_epoch = 0
model_template = UNet(num_class).to(device)
model_source = UNet(num_class).to(device)
model_corr2softmax = Corr2Softmax(200., 0.).to(device)
model_trans_template = UNet(num_class).to(device)
model_trans_source = UNet(num_class).to(device)
model_trans_corr2softmax = Corr2Softmax(11.72, 0.).to(device)


optimizer_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=4e-3)
optimizer_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=4e-3)
optimizer_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)
optimizer_trans_ft_temp = optim.AdamW(filter(lambda p: p.requires_grad, model_trans_template.parameters()), lr=4e-3)
optimizer_trans_ft_src = optim.AdamW(filter(lambda p: p.requires_grad, model_trans_source.parameters()), lr=4e-3)
optimizer_trans_c2s = optim.AdamW(filter(lambda p: p.requires_grad, model_trans_corr2softmax.parameters()), lr=5e-2)

exp_lr_scheduler_temp = lr_scheduler.StepLR(optimizer_ft_temp, step_size=1, gamma=0.8)
exp_lr_scheduler_src = lr_scheduler.StepLR(optimizer_ft_src, step_size=1, gamma=0.8)
exp_lr_scheduler_trans_temp = lr_scheduler.StepLR(optimizer_trans_ft_temp, step_size=1, gamma=0.8)
exp_lr_scheduler_trans_src = lr_scheduler.StepLR(optimizer_trans_ft_src, step_size=1, gamma=0.8)


# load pretrained model based on the input pretrained mode
if load_pretrained:
    if load_pretrained_mode == 'all':
        if load_optimizer:
            model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
            optimizer_ft_temp, optimizer_ft_src, optimizer_c2s,  optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s,\
                start_epoch = load_checkpoint(\
                                            checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                            optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)
        else:
            model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
            _, _, _, _, _, _,\
                start_epoch = load_checkpoint(\
                                            checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                            optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)

    if load_pretrained_mode == 'trans':
        model_trans_template, model_trans_source,\
            start_epoch = load_trans_checkpoint(\
                                        checkpoint_path, model_trans_template, model_trans_source,\
                                        device)
    if load_pretrained_mode == 'rot':
        model_template, model_source, model_corr2softmax,\
        optimizer_ft_temp, optimizer_ft_src, optimizer_c2s = load_rot_checkpoint(\
                                        checkpoint_path, model_template, model_source, model_corr2softmax,\
                                        optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, device)

model_template, model_source = train_model(model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                            optimizer_ft_temp, optimizer_ft_src, optimizer_c2s,  optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s,\
                                            exp_lr_scheduler_temp, exp_lr_scheduler_src, exp_lr_scheduler_trans_temp, exp_lr_scheduler_trans_src,\
                                            save_path, start_epoch, num_epochs=700)
