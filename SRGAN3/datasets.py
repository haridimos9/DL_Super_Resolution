import glob
import random
import os
import numpy as np


import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

mean_hr = np.array([0.5, 0.5, 0.5])
std_hr = np.array([0.5, 0.5, 0.5])

mean_lr = np.array([0,0,0])
std_lr = np.array([1,1,1])

class ImageDataset(Dataset):
    def __init__(self, root, hr_height, hr_width, downsampling, frames_per_sample, mode = 'train'):

        self.root = root

        with open(root+ f'/{mode}.txt', 'r') as f:
            self.lines = f.readlines()


        # img_ = Image.open(root +'/'+self.lines[0].strip() + '/im1.png')
        # hr_width, hr_height = img_.size

        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.CenterCrop((hr_height,hr_width)),
                transforms.Resize((hr_height // downsampling,
                                  hr_width // downsampling), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean_lr, std_lr),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.CenterCrop((hr_height,hr_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean_hr, std_hr),
            ]
        )

        self.frames_per_sample = frames_per_sample

    def __getitem__(self, index):

        dir_name = self.root + '/' + self.lines[index].strip()

        images_hr = []
        images_lr = []
        for img_idx in list(range(self.frames_per_sample)):
            img = Image.open(dir_name + f'/im{img_idx+1}.png')
            img_lr = self.lr_transform(img)
            images_lr.append(img_lr)

            img_hr = self.hr_transform(img)
            images_hr.append(img_hr)

        images_lr = torch.stack(images_lr).permute(1, 0, 2, 3)  # C, D, H, W
        images_hr = torch.stack(images_hr).permute(1, 0, 2, 3)

        return {"lr": images_lr, "hr": images_hr}

    def __len__(self):
        return len(self.lines)



class TestDataset(Dataset):
    def __init__(self, infer_dir):
        self.files = sorted(glob.glob(infer_dir+'/*.png'))

        self.transform = nn.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean_lr, std_lr)   
            ]
        )


    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)

        return {'lr':img}

    def __len__(self):
        return len(self.files)
