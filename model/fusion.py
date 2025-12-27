import torch.nn as nn
import functools
from utils import get_uperleft_denominator
from .module_util import *
import torch.nn as nn
from .sft import SFT_Layer

class SFT_Layer(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, nf, kernel_size=1, stride=1, padding=0)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.add_conv1 = nn.Conv2d(para + nf, nf, kernel_size=1, stride=1, padding=0)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class AlignedModule(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(outplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)
        self.conv_last = nn.Conv2d(inplane, outplane, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, h_feature, low_feature, upsample):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        if upsample:
            h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        h_feature = self.conv_last(h_feature)
        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x, fre=False):
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        atten = torch.sigmoid(atten)
        return feat*atten

class fusion_SFT(nn.Module):
    ''' SFT layer '''
    def __init__(self, nf=64, para=64):
        super(fusion_SFT, self).__init__()
        self.fusion_conv = nn.Sequential(*[
            nn.Conv2d(2*nf+2*para, para, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        ])
        self.mul_conv1 = nn.Conv2d(nf+para, nf, kernel_size=1, stride=1, padding=0)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.add_conv1 = nn.Conv2d(nf + para, nf, kernel_size=1, stride=1, padding=0)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.align_depth = SFT_Layer(nf, para)
        self.align_seg = SFT_Layer(nf, para)
    def forward(self, x):
        kernel_depth = self.align_depth(x[0], x[1])  
        kernel_seg = self.align_seg(x[0], x[2])  
        fusion = torch.cat((kernel_depth, kernel_seg, x[1], x[2]), dim=1).contiguous()
        fusion = self.fusion_conv(fusion)
        cat_input = torch.cat((x[0], fusion), dim=1).contiguous()
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
class fusion(nn.Module):
    def __init__(self, nf=64, nf_para=64 ,gc=32, bias=True,nb=2):
        super(fusion, self).__init__()
        self.fusion = fusion_SFT(nf, nf_para)
        self.body = nn.Sequential(*[ResidualDenseBlock_5C(nf=nf, gc=nf//2) for _ in range(nb)])
    def forward(self, x):
        out = self.fusion(x)
        out = self.body(out)
        return [out*0.2 + x[0], x[1], x[2]]