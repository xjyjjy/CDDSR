import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import time 

from . import register, models
from utils import make_coord
from utils import show_feature_map
from utils import save_img
from utils import make_serise, make_serise_clip

import os
import torch
import torchvision.utils as vutils

from .PE_head import PE_head
#from blind_model_2_1.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
@register('s1') # to reduce parameters
class OUR(nn.Module):

    def __init__(self, encoder_spec, kernel_encoder_spec, imnet_spec, restore_spec, feat_unfold = False, local_ensemble = False,
                 cell_decode = False, width=256, blocks=16, code_length=15, kernel_size = 21, pca_matrix_path = None):
        super().__init__()
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        self.restore = models.make(restore_spec)
        self.code_length = code_length
        self.kernel_size = 21
        self.width = width
        nf = 64
        code_length = 15
        self.dep_kernel = PE_head(nf=nf)
        self.seg_kernel = PE_head(nf=nf)
        if  kernel_encoder_spec is not None:
            self.kernel_enccoder = models.make(kernel_encoder_spec)
        else:
            self.kernel_enccoder = None
        self.encoder = models.make(encoder_spec)
        self.imnet = models.make(imnet_spec)


    def forward(self, inp, input_depth, input_seg, coord, scale, cell, gt, kernel_gt, train_stage="train", epoch=1000):

        inp_shape = inp.shape
        bs, c, h, w = inp_shape
        kernel = kernel_gt
        shape = coord.shape
        coord_scale = coord.view(bs,-1,2)
        cell_scale = (cell).unsqueeze(1).repeat(1, coord_scale.shape[1], 1)
        input_depth = self.dep_kernel(input_depth)  
        input_seg = self.seg_kernel(input_seg)

        lr_feature = self.encoder(inp)
        lr_feature = self.imnet(lr_feature, coord_scale, cell_scale, lr_feature.device).view(bs, *shape[1:3],-1).permute(0,3,1,2).contiguous()
        if self.kernel_enccoder is not None:
            kernel_feature = self.kernel_enccoder(kernel)
        else:
            kernel_feature = kernel
        image_feature, kernel = self.restore(kernel_feature, input_depth, input_seg, lr_feature, coord_scale, cell_scale, inp)
        return image_feature ,kernel, None


@register('s2')
class OUR(nn.Module):
    def __init__(self, scale_encoder_spec, kernel_esti_spec, imnet_spec, restore_spec, encoder_spec, feat_unfold = False, local_ensemble = False,
                 cell_decode = False, width=256, blocks=16, code_length=15, kernel_size = 21, pca_matrix_path = None, loop_time = 3):
        super().__init__()
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        # self.scale_encoder =  models.make(scale_encoder_spec)
        nf = 64
        self.dep_kernel = PE_head(nf=nf)
        self.seg_kernel = PE_head(nf=nf)
        self.kernel_esti = models.make(kernel_esti_spec)
        self.restore = models.make(restore_spec)
        self.code_length = code_length
        self.kernel_size = 21
        self.width = width
        self.loop_time = loop_time
        nf = 64
        code_length = 15
        self.encoder = models.make(encoder_spec)
        self.imnet = models.make(imnet_spec)
        
    def save_feature_channel_to_dir(self, feat, out_root, a):
        feat = feat.detach().cpu()
        B, C, H, W = feat.shape

        for c in range(C):
            channel_dir = os.path.join(out_root, str(c))
            os.makedirs(channel_dir, exist_ok=True)

            for b in range(B):
                x = feat[b, c]
                x_min = x.min()
                x_max = x.max()
                x_norm = (x - x_min) / (x_max - x_min + 1e-6)

                save_path = os.path.join(channel_dir, f"a{a:02d}_b{b:02d}.png")
                vutils.save_image(x_norm.unsqueeze(0), save_path)
    def forward(self, inp, input_depth, input_seg, coord, scale, cell, gt, kernel_gt, train_stage="train", epoch=0):


        inp_shape = inp.shape
        bs, c, h, w = inp_shape
        kernel = kernel_gt
        shape = coord.shape
        coord_scale = coord.view(bs,-1,2)
        cell_scale = (cell).unsqueeze(1).repeat(1, coord_scale.shape[1], 1)

        image_feature = F.grid_sample(inp, coord_scale.flip(-1).unsqueeze(1), mode='bicubic',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                     .permute(0, 2, 1).view(bs, *shape[1:3], 3).permute(0,3,1,2).contiguous()
        
        
        # print(train_stage)
        if epoch < 300:
            self.restore.eval()
            self.dep_kernel.eval()
            self.seg_kernel.eval()
            self.encoder.eval()
            self.imnet.eval()
            self.kernel_esti.train()
            with torch.no_grad():
                input_depth = self.dep_kernel(input_depth).detach()  
                input_seg = self.seg_kernel(input_seg).detach()
                lr_feature = self.encoder(inp).detach()
                lr_feature_ = self.imnet(lr_feature, coord_scale, cell_scale, lr_feature.device).view(bs, *shape[1:3],-1).permute(0,3,1,2).contiguous().detach()
        else:
            if 'train' in train_stage:
                self.restore.train()
                self.kernel_esti.train()
                self.dep_kernel.train()
                self.seg_kernel.train()
                self.encoder.train()
                self.imnet.train()
                lr_feature = self.encoder(inp)
                lr_feature_ = self.imnet(lr_feature, coord_scale, cell_scale, lr_feature.device).view(bs, *shape[1:3],-1).permute(0,3,1,2).contiguous()
                input_depth = self.dep_kernel(input_depth)  
                input_seg = self.seg_kernel(input_seg)
            elif train_stage=='bench':
                self.restore.train()
                self.dep_kernel.eval()
                self.seg_kernel.eval()
                self.encoder.eval()
                self.imnet.train()
                self.kernel_esti.eval()
                with torch.no_grad():
                    lr_feature = self.encoder(inp)
                    input_depth = self.dep_kernel(input_depth)  
                    input_seg = self.seg_kernel(input_seg)
                lr_feature_ = self.imnet(lr_feature, coord_scale, cell_scale, lr_feature.device).view(bs, *shape[1:3],-1).permute(0,3,1,2).contiguous()
            else:
                with torch.no_grad():
                    input_depth = self.dep_kernel(input_depth)  
                    input_seg = self.seg_kernel(input_seg)
                    lr_feature = self.encoder(inp)
                    lr_feature_ = self.imnet(lr_feature, coord_scale, cell_scale, lr_feature.device).view(bs, *shape[1:3],-1).permute(0,3,1,2).contiguous()
        a = random.randint(1, 100000)
        self.save_feature_channel_to_dir(input_depth, "/gdata2/xiajy/output/prior_visual/depth/", a)
        self.save_feature_channel_to_dir(input_seg, "/gdata2/xiajy/output/prior_visual/seg/", a)
        imgs = []
        ker_maps = []
        # self.loop_time = 4
        for i in range(self.loop_time):
            if isinstance(image_feature, list):
                image_feature = image_feature[0]

            if epoch < 300:
                kernel = self.kernel_esti(lr_feature, input_depth, input_seg, image_feature.detach(), shape[1:3],coord_scale, cell)
                with torch.no_grad():
                    image_feature, _ = self.restore(kernel.detach(), input_depth, input_seg, lr_feature_, coord_scale, cell_scale, inp)
            else:
                if train_stage == 'bench':
                    with torch.no_grad():
                        kernel = self.kernel_esti(lr_feature, input_depth, input_seg, image_feature.detach(), shape[1:3], coord_scale, cell)
                    image_feature, _ = self.restore(kernel.detach(), input_depth, input_seg, lr_feature_, coord_scale, cell_scale, inp)
                else:
                    kernel = self.kernel_esti(lr_feature, input_depth, input_seg, image_feature.detach(), shape[1:3], coord_scale, cell)
                    image_feature, _ = self.restore(kernel.detach(), input_depth, input_seg, lr_feature_, coord_scale, cell_scale, inp)

            if train_stage == "stage3_test":
                imgs.append(image_feature)
                ker_maps.append(kernel)


        if train_stage == "stage3_test":
            return imgs, ker_maps, None
        else:
            return image_feature ,kernel, None