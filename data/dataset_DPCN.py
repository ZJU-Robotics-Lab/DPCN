from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import torch
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(".."))
from utils.utils import *
from data.data_utils import *

# datacsv = pd.read_csv("./data_train/ground_truth.csv")

# template_list = datacsv["name"].values  
# template_list = [i+".jpg" for i in template_list]
# template_list = [os.path.join("./data_train/ground/",i) for i in template_list ]
# template_train_list = template_list[:5000]
# template_val_list = template_list[:5000]

# source_list = datacsv["name"].values
# source_list = [i+".jpg" for i in source_list]
# source_list = [os.path.join("./data_train/aerial/",i) for i in source_list ]
# source_train_list = source_list[:5000]
# source_val_list = source_list[:5000]

# ground_truth_list = datacsv["rotation"].values
# ground_truth_train_list = ground_truth_list[5000:]
# ground_truth_val_list = ground_truth_list[:5000]
# ground_truth_train_list = torch.from_numpy(ground_truth_train_list)
# ground_truth_val_list = torch.from_numpy(ground_truth_val_list)

# trans = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
# ])


class AeroGroundDataset_train(Dataset):
    def __init__(self, template_train_list, source_train_list, gt_rot_train_list, gt_scale_train_list, gt_x_train_list, gt_y_train_list, loader=default_loader):
        self.template_path_list = template_train_list #this is a list of the path to the template image
        self.source_path_list = source_train_list 
        self.gt_rot_list = gt_rot_train_list
        self.gt_scale_list = gt_scale_train_list
        self.loader = loader
        self.gt_trans_x = gt_x_train_list
        self.gt_trans_y = gt_y_train_list

        # add x y theta
    def __len__(self):
        return len(self.template_path_list)

    def __getitem__(self, index):
        # print(np.shape(this_source))
        this_template_path = self.template_path_list[index]
        this_source_path = self.source_path_list[index]
        rot_gt = self.gt_rot_list[index]
        scale_gt = self.gt_scale_list[index]
        trans_x = self.gt_trans_x[index]
        trans_y = self.gt_trans_y[index]

        this_template, _, _, _, _,_ = self.loader(this_template_path, resize_shape=256)
        this_source, _, _, scaling_factor, h_original, w_original = self.loader(this_source_path, resize_shape=256, change_scale=True)
        # print("this gt =", rot_gt)
        # gt_tensor = get_gt_tensor(rot_gt, this_template.size(1))
        # print("gt_tensor =", gt_tensor)
        # rot_gt += angle_t + angle_s
        # rot_gt += 180.
        # rot_gt %= 360
        # rot_gt -= 180.
        rot_gt = torch.tensor(rot_gt)
        scale_gt = torch.tensor(scale_gt) * scaling_factor

        trans = np.array((trans_y/(h_original/this_template.size(1)), trans_x/(w_original/this_template.size(1))))
        
        gt_trans = torch.tensor(trans)
        # add x y theta
        return [this_template, this_source, rot_gt, scale_gt, gt_trans]

class AeroGroundDataset_val(Dataset):
    def __init__(self, template_val_list, source_val_list, gt_rot_val_list, gt_scale_val_list, gt_x_val_list, gt_y_val_list, loader=default_loader):
        self.template_path_list = template_val_list #this is a list of the path to the template image
        self.source_path_list = source_val_list 
        self.gt_rot_list = gt_rot_val_list
        self.gt_scale_list = gt_scale_val_list
        self.loader = loader
        self.gt_trans_x = gt_x_val_list
        self.gt_trans_y = gt_y_val_list

    def __len__(self):
        return len(self.template_path_list)

    def __getitem__(self, index):
        
        this_template_path = self.template_path_list[index]
        this_source_path = self.source_path_list[index]
        rot_gt = self.gt_rot_list[index]
        scale_gt = self.gt_scale_list[index]
        trans_x = self.gt_trans_x[index]
        trans_y = self.gt_trans_y[index]
        this_template, _, _, _, _, _ = self.loader(this_template_path, resize_shape=256)
        this_source, _, _, scaling_factor, h_original, w_original = self.loader(this_source_path, resize_shape=256, change_scale=True)

        # gt_tensor = get_gt_tensor(rot_gt, this_template.size(1))

        # rot_gt += angle_t + angle_s
        # rot_gt += 180.
        # rot_gt %= 360
        # rot_gt -= 180.
        rot_gt = torch.tensor(rot_gt)
        scale_gt = torch.tensor(scale_gt) * scaling_factor

        # trans_x = (torch.sign(-trans_x) + 1) / 2 * 256 + trans_x
        # trans_y = (torch.sign(-trans_y) + 1) / 2 * 256 + trans_y
        trans = np.array((trans_y/(h_original/this_template.size(1)), trans_x/(w_original/this_template.size(1))))
        gt_trans = torch.tensor(trans)

        return [this_template, this_source, rot_gt, scale_gt, gt_trans]


def DPCNdataloader(batch_size):
    # use the same transformations for train/val in this example
    path = "./data"
    datacsv = pd.read_csv(path + "/data_train_qsdjt_stereo_sat/ground_truth_qsdjt_lidar.csv")
    train_upper = 6000
    val_num = 2000
    val_upper = train_upper+val_num

    template_list = datacsv["name"].values
    template_list = [i+".jpg" for i in template_list]
    template_list = [os.path.join(path + "/data_train_qsdjt_stereo_sat/ground/",i) for i in template_list ]
    template_train_list = template_list[:train_upper]
    template_val_list = template_list[train_upper:val_upper]

    source_list = datacsv["name"].values
    source_list = [i+".jpg" for i in source_list]
    source_list = [os.path.join(path + "/data_train_qsdjt_stereo_sat/aerial/",i) for i in source_list ]
    source_train_list = source_list[:train_upper]
    source_val_list = source_list[train_upper:val_upper]

    gt_rot_list = datacsv["rotation"].values
    gt_rot_train_list = gt_rot_list[:train_upper]
    gt_rot_val_list = gt_rot_list[train_upper:val_upper]
    gt_rot_train_list = torch.from_numpy(gt_rot_train_list)
    gt_rot_val_list = torch.from_numpy(gt_rot_val_list)

    gt_scale_list = datacsv["rotation"].values * 0 +1.0
    gt_scale_train_list = gt_scale_list[:train_upper]
    gt_scale_val_list = gt_scale_list[train_upper:val_upper]
    gt_scale_train_list = torch.from_numpy(gt_scale_train_list)
    gt_scale_val_list = torch.from_numpy(gt_scale_val_list)

    gt_x_list = datacsv["shift_x"].values
    gt_x_train_list = gt_x_list[:train_upper]
    gt_x_val_list = gt_x_list[train_upper:val_upper]
    gt_x_train_list = torch.from_numpy(gt_x_train_list)
    gt_x_val_list = torch.from_numpy(gt_x_val_list)

    gt_y_list = datacsv["shift_y"].values
    gt_y_train_list = gt_y_list[:train_upper]
    gt_y_val_list = gt_y_list[train_upper:val_upper]
    gt_y_train_list = torch.from_numpy(gt_y_train_list)
    gt_y_val_list = torch.from_numpy(gt_y_val_list)

    train_set = AeroGroundDataset_train(template_train_list, source_train_list, gt_rot_train_list, gt_scale_train_list, gt_x_train_list, gt_y_train_list)
    val_set = AeroGroundDataset_val(template_val_list, source_val_list, gt_rot_val_list, gt_scale_val_list, gt_x_val_list, gt_y_val_list)

    image_datasets = {
        'train': train_set, 'val': val_set
    }


    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    return dataloaders
