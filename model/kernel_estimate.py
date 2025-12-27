
import torch.nn as nn
import functools
from .module_util import *

from argparse import Namespace
from . import register, models

from .fusion import fusion
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x
class RRDB_Block(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB_Block, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
    def forward(self, input):
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + input
@register('kernel_estimate')
class KernelEstimate(nn.Module):
    def __init__(self, in_nc0, in_nc1, nf, out_nc, nb, mlp_spec, encoder_spec, kernel_size, hr_feature=128, sm_para=64):
        self.in_nc0 = in_nc0
        self.in_nc1 = in_nc1
        self.nf = nf
        self.kernel_size = kernel_size
        super().__init__()
        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, out_nc, 3, 1, 1),
            nn.Tanh()
        )
        self.scale0_lr = models.make(mlp_spec, args={'in_dim': 9*64 + 2, 'out_dim': nf})
        self.scale0 = nn.Sequential(
            *[fusion(nf, nf, nb=4) for _ in range(nb)]
        )
        self.encoder_hr = models.make(encoder_spec, args={'n_feats': hr_feature})
        self.encoder_dep_0 = nn.Sequential(*[
            nn.Conv2d(sm_para, nf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            *[RRDB_Block(nf=nf,gc=nf//2) for _ in range(1)],
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
        ])
        self.encoder_seg_0 = nn.Sequential(*[
            nn.Conv2d(sm_para, nf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            *[RRDB_Block(nf=nf,gc=nf//2) for _ in range(1)],
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1)
        ])
       
    def forward(self, lr, depth, seg, hr, shape, coord, cell): # lr hr  5.1.9.2
        device = torch.cuda.current_device()
        hr_feature = self.encoder_hr(hr)
        bs, C, H, W = hr.shape
        bs, c, h, w= lr.shape
        feature_residual = self.scale0_lr(lr, hr_feature, coord,cell, device = lr.device).view(bs,H, W, -1).permute(0,3,1,2).contiguous()
        depth_t = self.encoder_dep_0(depth)
        seg_t = self.encoder_seg_0(seg)
        feature, _, _= self.scale0([feature_residual, depth_t, seg_t])
        f = self.tail(feature + feature_residual)
        return f