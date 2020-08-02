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
from validate import val_model
def train_rot_scale(template_rot, source_rot, groundTruth_rot, groundTruth_scale, model_template_rot, model_source_rot, model_corr2softmax_rot, phase, device):
    print("                             ")
    print("                             TRAINING ROTATION AND SCALE")
    print("                             ")
    
    with torch.set_grad_enabled(phase == 'train'):
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
        source_logpolar.retain_grad()
        phase_corr_layer_rs = PhaseCorr(device, logbase_rot, model_corr2softmax_rot)
        rotation_cal, scale_cal, softmax_result_rot, corr_result_rot = phase_corr_layer_rs(template_logpolar, source_logpolar)
        # print("logbaseeeeeeee", logbase_rot)
# use softmax

        # softmax_final = softmax_result.clone()

        # for batch_num in range(softmax_result.shape[0]):
        #     lower, upper = softmax_result[batch_num].clone().chunk(2,0)
        #     if success[batch_num] == 1:
        #         softmax_final[batch_num] = torch.cat((upper,lower),0)
        #     else:
        #         softmax_final[batch_num] = softmax_result[batch_num].clone()

        # softmax_final = torch.sum(softmax_final, 2, keepdim=False)



# use phasecorr result

        corr_final_rot = corr_result_rot.clone()
        corr_visual_rot = corr_final_rot.unsqueeze(-1)
        corr_visual_rot = corr_visual_rot.permute(0,3,1,2)
                
        corr_final_rot = torch.sum(corr_final_rot, 2, keepdim=False)

        corr_final_rot = model_corr2softmax_rot(corr_final_rot)
        input_rot = nn.functional.softmax(corr_final_rot.clone(), dim=-1)
        indice_rot = np.linspace(0, 1, 256)
        indice_rot = torch.tensor(np.reshape(indice_rot, (-1, 256))).to(device)
        rot_exp = torch.sum((256 - 1) * input_rot * indice_rot, dim=-1)        
                
        # groundTruth = groundTruth.to(device)
        



        corr_final_scale = corr_result_rot.clone()
        corr_final_scale = torch.sum(corr_final_scale,1,keepdim=False)
        corr_final_scale = model_corr2softmax_rot(corr_final_scale)

# consider angle and scale as the loss 

        groundTruth_rot = groundTruth_rot.to(device)
        gt_number = groundTruth_rot.clone()
        gt_angle = GT_angle_convert(gt_number,256)
        gt_angle = gt_angle.to(device)

        groundTruth_scale = groundTruth_scale.to(device)
        gt_scale = GT_scale_convert(groundTruth_scale.clone(), logbase_rot, 256)
        gt_scale = gt_scale.to(device)



        

        ACC_rot = (1-(rotation_cal-groundTruth_rot).abs()/(groundTruth_rot+0.00000000000000001)).mean()
        if ACC_rot <= 0:
            ACC_rot = torch.Tensor([0.5])
        ACC_scale = (1-(scale_cal-groundTruth_scale).abs()/(groundTruth_scale+0.00000000000000001)).mean()
        if ACC_scale <= 0:
            ACC_scale = torch.Tensor([0.5])




        print("rotation =", rotation_cal)

        print("gt_rot =", groundTruth_rot, "\n")
        print("scale =", scale_cal)
        print("gt_scale =", groundTruth_scale, "\n")
        
        # print("gt_angle ", gt_angle)
        print("ACC_rot = ",ACC_rot.item()*100,"%")
        print("ACC_scale = ",ACC_scale.item()*100,"%")


# # flatten the tensor:
        # b_loss,h_loss,w_loss = groundTruth.shape
        # groundTruth = groundTruth.reshape(b_loss,h_loss*w_loss)
        # softmax_final = softmax_final.reshape(b_loss,h_loss*w_loss)


# set the loss function:
        # compute_loss = torch.nn.KLDivLoss(reduction="sum").to(device)
        # compute_loss = torch.nn.BCEWithLogitsLoss(reduction="sum").to(device)
        compute_loss_rot = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
        compute_loss_scale = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
        compute_mse = torch.nn.MSELoss()
        compute_l1=torch.nn.L1Loss().to(device)
        # compute_loss = torch.nn.MSELoss()
        # compute_loss=torch.nn.L1Loss()

        loss_rot = compute_loss_rot(corr_final_rot,gt_angle)
        loss_scale = compute_loss_scale(corr_final_scale,gt_scale)
        loss_l1_rot = compute_l1(rot_exp, groundTruth_rot)
        loss_l1_scale = compute_l1(scale_cal, groundTruth_scale)
        loss_mse_rot = compute_mse(rotation_cal, groundTruth_rot)
        loss_mse_scale = compute_mse(scale_cal, groundTruth_scale)
        print("loss rot ==", loss_rot)
        print("loss scale ==", loss_scale)
        return loss_rot, loss_scale, loss_l1_rot, loss_mse_rot, loss_l1_scale, loss_mse_scale, template_visual_rot, source_visual_rot

def train_translation(template_trans, source_trans, groundTruth_number, scale_gt, gt_trans, model_template_trans, model_source_trans, model_corr2softmax_trans, phase, dsnt, device ):
    print("                             ")
    print("                             TRAINING TRANSLATION")
    print("                             ")
    with torch.set_grad_enabled(phase == 'train'):
# # for toy dataset
#         b, c, h, w = source_trans.shape
#         center = torch.ones(b,2).to(device)
#         center[:, 0] = h // 2
#         center[:, 1] = w // 2
#         angle_rot = torch.ones(b).to(device) * (-groundTruth_number.to(device))
#         scale_rot = torch.ones(b).to(device) 
#         rot_mat = kornia.get_rotation_matrix2d(center, angle_rot, scale_rot)
#         source_trans = kornia.warp_affine(source_trans.to(device), rot_mat, dsize=(h, w))

# for AGDatase
        b, c, h, w = source_trans.shape
        center = torch.ones(b,2).to(device)
        center[:, 0] = h // 2
        center[:, 1] = w // 2
        angle_rot = torch.ones(b).to(device) * (-groundTruth_number.to(device))
        scale_rot = torch.ones(b).to(device) / scale_gt.to(device)
        rot_mat = kornia.get_rotation_matrix2d(center, angle_rot, scale_rot)
        source_trans = kornia.warp_affine(source_trans.to(device), rot_mat, dsize=(h, w))


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
        phase_corr_layer_xy = PhaseCorr(device, logbase_trans, model_corr2softmax_trans, trans=True)
        t0, t1, softmax_result_trans, corr_result_trans = phase_corr_layer_xy(template_unet_trans.to(device), source_unet_trans.to(device))

# use phasecorr result
        if not dsnt:

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
            tranformation_y = torch.sum((256 - 1) * input_c * indices_c, dim=-1)
            # tranformation_y = torch.argmax(corr_y, dim=-1)

            corr_x = torch.sum(corr_final_trans.clone(), 1, keepdim=False)
            # corr_final_trans = corr_final_trans.reshape(b, h*w)
            corr_x = model_corr2softmax_trans(corr_x)
            input_r = nn.functional.softmax(corr_x.clone(), dim=-1)
            indices_r = np.linspace(0, 1, 256)
            indices_r = torch.tensor(np.reshape(indices_r, (-1, 256))).to(device)
            # tranformation_x = torch.argmax(corr_x, dim=-1)
            tranformation_x = torch.sum((256 - 1) * input_r * indices_r, dim=-1)

    # only consider angle as the los
            # softmax_result = torch.sum(corr_result.clone(), 2, keepdim=False)
            # softmax_final = softmax_result.clone()
            # # softmax_visual = softmax_final.unsqueeze(-1)
            # # softmax_visual = softmax_visual.permute(0,3,1,2)
                    
            # softmax_final = softmax_final.reshape(b, h*w)
            # softmax_final = model_corr2softmax(softmax_final.clone())
            gt_trans_orig = gt_trans.clone().to(device)

            # print("err_true = ",err_true.item()*100,"%")

            gt_trans_convert = GT_trans_convert(gt_trans_orig, [256, 256])
            gt_trans_convert_y = gt_trans_convert[:,0]
            gt_trans_convert_x = gt_trans_convert[:,1]

            print("trans x", tranformation_x)
            print("gt_convert x", gt_trans_convert_x, "\n")


            print("trans y", tranformation_y)
            print("gt_convert y", gt_trans_convert_y,"\n")



    # set the loss function:
            # compute_loss = torch.nn.KLDivLoss(reduction="sum").to(device)
            # compute_loss = torch.nn.BCEWithLogitsLoss(reduction="sum").to(device)
            compute_loss_y = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
            compute_loss_x = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
            # mse_loss = torch.nn.MSELoss(reduce=True)
            # compute_loss = torch.nn.NLLLoss()
            # compute_l1loss_a = torch.nn.L1Loss()
            # compute_l1loss_x = torch.nn.L1Loss()
            compute_mse = torch.nn.MSELoss()
            compute_l1=torch.nn.L1Loss()

            # mse_loss = mse_loss(rotation_cal.float(), groundTruth_number.float())
            loss_l1_x = compute_l1(tranformation_x, gt_trans_convert_x)
            loss_l1_y = compute_l1(tranformation_y, gt_trans_convert_y)
            loss_mse_x = compute_mse(tranformation_x, gt_trans_convert_x)
            loss_mse_y = compute_mse(tranformation_y, gt_trans_convert_y)
            loss_x = compute_loss_x(corr_x, gt_trans_convert_x)
            loss_y = compute_loss_y(corr_y, gt_trans_convert_y)
            total_loss = loss_x + loss_y + loss_l1_x + loss_l1_y
            return loss_y, loss_x, total_loss, loss_l1_x,loss_l1_y,loss_mse_x, loss_mse_y, template_visual_trans, source_visual_trans
        else:
            corr_final_trans = corr_result_trans.clone()
            # corr_visual = corr_final_trans.unsqueeze(-1)
            # corr_visual = corr_visual.permute(0,3,1,2)
            corr_y = torch.sum(corr_final_trans.clone(), 2, keepdim=False)
            # corr_2d = corr_final_trans.clone().reshape(b, h*w)
            # corr_2d = model_corr2softmax(corr_2d)
            corr_y = model_corr2softmax_trans(corr_y)

            corr_x = torch.sum(corr_final_trans.clone(), 1, keepdim=False)
            # corr_final_trans = corr_final_trans.reshape(b, h*w)
            corr_x = model_corr2softmax_trans(corr_x)

            corr_mat_dsnt_trans = corr_result_trans.clone().unsqueeze(-1)
            corr_mat_dsnt_trans_final = model_corr2softmax_trans(corr_mat_dsnt_trans)
            corr_mat_dsnt_trans_final = kornia.spatial_softmax2d(corr_mat_dsnt_trans_final)
            coors_trans = kornia.spatial_expectation2d(corr_mat_dsnt_trans_final,False)
            tranformation_x = coors_trans[:,0,0]
            tranformation_y = coors_trans[:,0,1]

    # only consider angle as the los
            # softmax_result = torch.sum(corr_result.clone(), 2, keepdim=False)
            # softmax_final = softmax_result.clone()
            # # softmax_visual = softmax_final.unsqueeze(-1)
            # # softmax_visual = softmax_visual.permute(0,3,1,2)
                    
            # softmax_final = softmax_final.reshape(b, h*w)
            # softmax_final = model_corr2softmax(softmax_final.clone())
            gt_trans_orig = gt_trans.clone().to(device)

            # print("err_true = ",err_true.item()*100,"%")

            gt_trans_convert = GT_trans_convert(gt_trans_orig, [256, 256])
            gt_trans_convert_y = gt_trans_convert[:,0]
            gt_trans_convert_x = gt_trans_convert[:,1]

            print("trans x", tranformation_x)
            print("gt_convert x", gt_trans_convert_x, "\n")


            print("trans y", tranformation_y)
            print("gt_convert y", gt_trans_convert_y,"\n")



    # set the loss function:
            # compute_loss = torch.nn.KLDivLoss(reduction="sum").to(device)
            # compute_loss = torch.nn.BCEWithLogitsLoss(reduction="sum").to(device)
            compute_loss_y = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
            compute_loss_x = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
            # mse_loss = torch.nn.MSELoss(reduce=True)
            # compute_loss = torch.nn.NLLLoss()
            # compute_l1loss_a = torch.nn.L1Loss()
            # compute_l1loss_x = torch.nn.L1Loss()
            compute_mse = torch.nn.MSELoss()
            compute_l1=torch.nn.L1Loss()

            # mse_loss = mse_loss(rotation_cal.float(), groundTruth_number.float())
            loss_l1_x = compute_l1(tranformation_x, gt_trans_convert_x)
            loss_l1_y = compute_l1(tranformation_y, gt_trans_convert_y)
            loss_mse_x = compute_mse(tranformation_x, gt_trans_convert_x.type(torch.FloatTensor).to(device))
            loss_mse_y = compute_mse(tranformation_y, gt_trans_convert_y.type(torch.FloatTensor).to(device))
            loss_x = compute_loss_x(corr_x, gt_trans_convert_x)
            loss_y = compute_loss_y(corr_y, gt_trans_convert_y)
            total_loss = 0.001*(loss_mse_x + loss_mse_y)
            return loss_y, loss_x, total_loss, loss_l1_x,loss_l1_y,loss_mse_x, loss_mse_y, template_visual_trans, source_visual_trans


