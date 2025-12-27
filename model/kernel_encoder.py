
import torch.nn as nn
import functools
from utils import get_uperleft_denominator
from .module_util import *

from argparse import Namespace
from . import register, models

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        torch.nn.init.normal_(self.conv1.weight, 0.0, 0.02)
        torch.nn.init.normal_(self.conv2.weight, 0.0, 0.02)
        torch.nn.init.normal_(self.conv3.weight, 0.0, 0.02)
        torch.nn.init.normal_(self.conv4.weight, 0.0, 0.02)
        torch.nn.init.normal_(self.conv5.weight, 0.0, 0.02)
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

@register('kernel_encoder')
class KernelEncoder(nn.Module):
    def __init__(self, nf, kernel_size, code_length, ch_div=4):
        self.kernel_size = kernel_size
        self.nf = nf
        super().__init__()
        
        self.kernel_conv = nn.Sequential(*[
            nn.Conv2d(kernel_size*kernel_size, nf//ch_div, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf//ch_div, nf, 1, 1, 0) 
        ])
        self.RRDB = nn.Sequential(*[RRDB_Block(nf, gc=nf//2)for _ in range(10)])
        self.kernel_conv_1 = nn.Sequential(*[
            nn.Conv2d(self.nf, self.nf//2, 3, 1, 1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.nf//2, code_length, 3, 1, 1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(code_length, code_length, 3, 1, 1), 
            nn.Tanh()
        ])
    def forward(self, kernel): 
        bs, c, h, w = kernel.shape
        kernl_information = self.kernel_conv(kernel)
        kernl_information = self.RRDB(kernl_information) + kernl_information
        kernl_information = self.kernel_conv_1(kernl_information)
        return kernl_information
        

        