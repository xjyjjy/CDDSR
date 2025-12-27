import functools
import random
import math
import time
import utils
#import cv2
from PIL import Image

import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
# from datasets.data_reshape import Dataset as Dataset_reshape
from torch.utils.data import Dataset as Dataset_1 # random patch size
# from datasets.data import Dataset as Dataset_1 # random patch
from torch.utils.data import Dataset

from torchvision import transforms
from utils import add_noise

from torchvision.utils import save_image as save_image_from_torch


from datasets import register
from utils import to_pixel_samples, to_pixel_kernel_samples, to_pixel_samples_grid, to_pixel_kernel_samples_grid, save_img, make_coord, tensor2img
import numpy as np

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))

class PCAEncoder(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight.double())
        self.size = self.weight.size()

    def forward(self, batch_kernel):
        B, HW = batch_kernel.size()  # [B, l, l]
        # print(self.weight)
        q =  self.weight.expand((B,) + self.size)
        print("Q",q.shape)
        return torch.bmm(
            batch_kernel.view((B, 1, HW)), self.weight.expand((B,) + self.size)
        ).view((B, -1))


@register('sr-blur-kernel') # depth no regulization
class SRBlurImplicitTriple(Dataset):
    def __init__(self, dataset, KernelWarehouse, pca_matrix, inp_size=None, hr_size=None, scale_min=1, scale_max=None,augment=False, sample_q=None,test=False, deg_type = 'depth'):
        self.dataset = dataset
        self.augment = augment
        self.hr_size = hr_size
        self.test = test
        self.kernel_warehouse = KernelWarehouse
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.inp_size = inp_size
        self.sample_q = sample_q
        self.code_length = 15
        self.deg_type = deg_type
        self.PCAencoder = PCAEncoder(pca_matrix[0])
        if test is False:
            self.hr_size = hr_size
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, scale=2, hr_size=64):
        img_hr, img_blur, depth, label, variance = self.dataset[idx]
        scale = random.uniform(self.scale_min, self.scale_max)
        kernel_warehouse = self.kernel_warehouse
        hr_size = self.hr_size
        depth = depth/255.
        if self.test is False:
            scale = scale #random.uniform(self.scale_min, self.scale_max)
        else:
            scale = self.scale_min
        if self.test is False:

            w_lr = self.inp_size
            w_hr = round(w_lr * scale)
            x0 = random.randint(0, img_hr.shape[-2] - w_hr)
            y0 = random.randint(0, img_hr.shape[-1] - w_hr)

            crop_hr = img_hr[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_blur = img_blur[:, x0: x0 + w_hr, y0: y0 + w_hr]
            variance = variance[x0: x0 + w_hr, y0: y0 + w_hr]
            crop_depth = depth[x0: x0 + w_hr, y0: y0 + w_hr]
            crop_label = label[x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_blur, w_lr)

        else:
            h_lr = math.floor(img_hr.shape[-2] / scale + 1e-9)
            w_lr = math.floor(img_hr.shape[-1] / scale + 1e-9)
            img_hr = img_hr[:, :round(h_lr * scale), :round(w_lr * scale)]
            variance = variance[:round(h_lr * scale), :round(w_lr * scale)] 
            crop_depth = depth[:round(h_lr * scale), :round(w_lr * scale)] 
            crop_label = label[:round(h_lr * scale), :round(w_lr * scale)] 
            img_blur = img_blur[:, :round(h_lr * scale), :round(w_lr * scale)]
            img_down = resize_fn(img_blur, (h_lr, w_lr))
            crop_blur, crop_lr, crop_hr = img_blur, img_down, img_hr

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5
            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            crop_depth = augment(crop_depth)
            crop_label = augment(crop_label)
            variance = augment(variance)
        
        hr_coord, hr_rgb = to_pixel_samples_grid(crop_hr.contiguous())
        if self.hr_size is not None:
            x1 = random.randint(0, hr_rgb.shape[0] - hr_size)
            y1 = random.randint(0, hr_rgb.shape[1] - hr_size)
            hr_rgb = hr_rgb[x1:x1+hr_size, y1:y1+hr_size,:]
            hr_coord = hr_coord[x1:x1+hr_size, y1:y1+hr_size,:]
            variance = variance[x1: x1 + hr_size, y1: y1 + hr_size]
            crop_depth = crop_depth[x1: x1 + hr_size, y1: y1 + hr_size]
            crop_label = crop_label[x1: x1 + hr_size, y1: y1 + hr_size]
        
        H ,W ,channel = hr_rgb.shape
        if self.test == False:
            if self.deg_type == 'depth':
                index = torch.round(variance.to(torch.float).squeeze() * 10000)%kernel_warehouse.shape[0]
                kernel_map = (kernel_warehouse[index.flatten().long()]).float().reshape(H,W,-1).permute(2,0,1)
            else:
                index = variance
                kernel_map = (kernel_warehouse[index.flatten().long()]).reshape(*index.shape,-1).permute(2,0,1)
                
        else:
            if self.deg_type == 'depth':
                index = torch.round(variance.to(torch.float).squeeze() * 10000)%kernel_warehouse.shape[0]
                kernel_map = (kernel_warehouse[index.flatten().long()]).float().reshape(H,W,-1).permute(2,0,1)
            else:
                index = variance
                kernel_map = (kernel_warehouse[index.flatten().long()]).reshape(*index.shape,-1).permute(2,0,1)
            crop_label = crop_label
            crop_depth =  crop_depth 
        H ,W ,channel = hr_rgb.shape
        hr_coord = hr_coord.contiguous()   
        hr_rgb = hr_rgb.contiguous()
        crop_label = crop_label.unsqueeze(0).contiguous()
        crop_depth = crop_depth.unsqueeze(0).contiguous()
        cell_for_liif = torch.ones_like(hr_coord.view(-1,2))
        cell_for_liif[:, 0] *= 2 / crop_hr.shape[-2]
        cell_for_liif[:, 1] *= 2 / crop_hr.shape[-1]
        coord_xy = [hr_coord[0, 0, 0], 
                 hr_coord[0, 0, 1], 
                 hr_coord[-1, -1, 0],
                 hr_coord[-1, -1, 1]]  
        coord_xy = torch.tensor(coord_xy)
        scale = torch.tensor(scale)
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        return {
            'inp': crop_lr,
            'depth': crop_depth.float(),
            'label': crop_label.float(),
            'kernel_map': kernel_map.float(),
            'cell_for_liif': cell_for_liif,
            'coord': hr_coord,
            'coord_xy': coord_xy,
            'scale': scale,
            'cell': cell,
            'gt': hr_rgb
        }
