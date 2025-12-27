import torch.nn as nn
import functools
from utils import get_uperleft_denominator
from .module_util import *
import torch.nn as nn
from argparse import Namespace
from . import register, models
class SFT_Layer(nn.Module):
    ''' SFT layer '''
    def __init__(self, nf=64, para=64):
        super(SFT_Layer, self).__init__()
        #self.deep_conv = OurBlock(nf, bias=True) #nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.mul_conv1 = nn.Conv2d(nf+para, nf, kernel_size=1, stride=1, padding=0)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.add_conv1 = nn.Conv2d(nf + para, nf, kernel_size=1, stride=1, padding=0)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        cat_input = torch.cat((x[0], x[1]), dim=1).contiguous()
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return x[0] * mul + add

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
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x
    

class RRDB_SFT(nn.Module):
    def __init__(self, nf, gc=32, para=15):
        super(RRDB_SFT, self).__init__()
        self.SFT = SFT_Layer(nf=nf, para=para)
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
    def forward(self, input):
        out = self.SFT([input[0], input[1]])
        out = self.RDB1(out)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return [out * 0.2 + input[0], input[1]]
    
class RRDB_fusion(nn.Module):
    def __init__(self, nf, gc=32, para=15):
        super(RRDB_fusion, self).__init__()
        self.SFT0 = SFT_Layer(nf=nf, para=para)
        self.SFT1 = SFT_Layer(nf=nf, para=para)
        self.SFT2 = SFT_Layer(nf=nf, para=para)
        self.conv0 = nn.Conv2d(para, para, 3, 1, 1)
        self.conv1 = nn.Conv2d(para, para, 3, 1, 1)
        self.conv2 = nn.Conv2d(para, para, 3, 1, 1)


        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

        
        self.RDB4 = ResidualDenseBlock_5C(nf, gc)
        self.RDB5 = ResidualDenseBlock_5C(nf, gc)
        self.RDB6 = ResidualDenseBlock_5C(nf, gc)

        
        self.RDB7 = ResidualDenseBlock_5C(nf, gc)
        self.RDB8 = ResidualDenseBlock_5C(nf, gc)
        self.RDB9 = ResidualDenseBlock_5C(nf, gc)

        self.RDB10 = ResidualDenseBlock_5C(nf, gc)
        self.RDB11 = ResidualDenseBlock_5C(nf, gc)
        self.RDB12 = ResidualDenseBlock_5C(nf, gc)
    def forward(self, input):
        input_1 = self.conv0(input[1]) 
        input_2 = self.conv1(input[2])
        input_3 = self.conv2(input[3])
        out = self.SFT0([input[0], input_1])
        out = self.RDB1(out)
        out = self.RDB2(out)
        out0 = self.RDB3(out) * 0.2 + input[0]
        
        out = self.SFT1([out0, input_2])
        out = self.RDB4(out)
        out = self.RDB5(out)
        out1 = self.RDB6(out) * 0.2 + out0

        out = self.SFT2([out1, input_3])
        out = self.RDB7(out)
        out = self.RDB8(out)
        out2 = self.RDB9(out) * 0.2 + out1
        
        out = self.RDB10(out2)
        out = self.RDB11(out)
        out = self.RDB12(out)
        out = out *0.2 + out2


        return [out, input[1], input[2], input[3]]

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
    
@register('restoration')
class Restore(nn.Module):
    def __init__(self, code_length_k, code_length, in_nc1, in_nc2, nf, out_nc, nb, win_size):
        self.code_length = code_length
        self.in_nc1 = in_nc1
        self.win_size = win_size
        # self.kernel_encoder_spec = kernel_encoder_spec
        self.nf = nf
        self.nb = nb
        super().__init__()
        self.ftype = 3
        self.head_kernel = nn.Sequential(*[
            nn.Conv2d(code_length_k, nf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, code_length, 3, stride=1, padding=1),
        ])
        self.dep_kernel = nn.Sequential(*[
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, code_length, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        ])
        self.seg_kernel = nn.Sequential(*[
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, code_length, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        ])
        self.body = nn.Sequential(*[RRDB_fusion(nf=nf,gc=nf//2, para=code_length)for _ in range(nb)])
        self.tail =nn.Sequential(*[
            nn.Conv2d(nf, out_nc, 3, stride=1, padding=1),
        ])

    def forward(self, kernel, input_depth, input_seg, lr_feature, coord_scale, cell_scale, lr_): #x0 kernel x1 image
        bs = kernel.shape[0]
        shape = kernel.shape[-2:]
        bs, C, H, W = kernel.shape

        kernel_feature = self.head_kernel(kernel)
        depth_feature = self.dep_kernel(input_depth)
        seg_feature = self.seg_kernel(input_seg)
        
        x = [lr_feature, kernel_feature, depth_feature, seg_feature]
        x = self.body(x)
        out = self.tail(x[0])
        out += F.grid_sample(lr_, coord_scale.flip(-1).unsqueeze(1), mode='bilinear',\
                       padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1).view(bs, *shape, -1).permute(0,3,1,2).contiguous()
        return out,  kernel
    
    
    