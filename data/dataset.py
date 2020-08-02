from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import torch
import sys
import data.simulation as simulation
from data.data_utils import *


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.rotate_image, self.gt_rot, self.gt_trans = simulation.generate_random_data(256, 256, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        rot = self.rotate_image[idx]
        gt_rot = self.gt_rot[idx]
        gt_trans = self.gt_trans[idx]
        gt_scale = torch.tensor(1.)
        
        if self.transform:
            image = self.transform(image)
            rot = self.transform(rot)
            gt_rots = torch.tensor(gt_rot)
            gt_scale = torch.tensor(1.)
            gt_trans = torch.tensor(gt_trans)
        # print("gt = ", gt)
        # print("gt tensor = ", gt_tensor)

        return [image, rot, gt_rots, gt_scale, gt_trans]

def generate_dataloader(batch_size):
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    train_set = SimDataset(2000, transform = trans)
    val_set = SimDataset(1000, transform = trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    return dataloaders