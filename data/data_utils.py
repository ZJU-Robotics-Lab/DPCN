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

def default_loader(path, resize_shape, change_scale = False):
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (h_original, w_original) = image.shape
    image = cv2.resize(image, dsize=(resize_shape,resize_shape), interpolation=cv2.INTER_CUBIC)
    angle = (np.random.rand()-0.5) * 0.
    angle += 180.
    angle %= 360
    angle -= 180.
    (h, w) = image.shape
    (cX, cY) = (w//2, h//2)

    t_x = np.random.rand() * 0.
    t_y = np.random.rand() * 0.
    translation = np.array((t_y, t_x))

    # arr = arr[0,]
    # rot = ndii.rotate(arr, angle)

    # N = np.float32([[1,0,t_x],[0,1,t_y]])
    # image = cv2.warpAffine(image, N, (w, h))
    # M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # image = cv2.warpAffine(image, M, (w, h))
    # image = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)

    np_image_data = np.asarray(image)
    image_tensor = trans(np_image_data)
    scaling_factor = 1
    if change_scale:
        center = torch.ones(1,2)
        center[:, 0] = h // 2
        center[:, 1] = w // 2
        scaling_factor = torch.tensor(np.random.rand()*0.2+1)
        angle_source = torch.ones(1) * 0.
        scale_source = torch.ones(1) * scaling_factor
        image_tensor = image_tensor.unsqueeze(0)
        rot_mat = kornia.get_rotation_matrix2d(center, angle_source, scale_source)
        image_tensor = kornia.warp_affine(image_tensor, rot_mat, dsize=(h, w))
        image_tensor = image_tensor.squeeze(0)
    # image = Image.open(path)
    # image = image.convert("1")
    # # image.show()
    # image = image.resize((128,128))
    # image_tensor = trans(image)
    return image_tensor, angle, translation, scaling_factor, h_original, w_original

def get_gt_tensor(this_gt, size):
    this_gt = this_gt +180
    gt_tensor_self = torch.zeros(size,size)
    angle_convert = this_gt*size/360
    angle_index = angle_convert//1 + (angle_convert%1+0.5)//1
    if angle_index.long() == size:
        angle_index = size-1
        gt_tensor_self[angle_index,0] = 1
    else:
        gt_tensor_self[angle_index.long(),0] = 1
    # print("angle_index", angle_index)

    return gt_tensor_self
