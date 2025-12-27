import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time 

from . import register, models
from utils import make_coord
from utils import show_feature_map
from utils import save_img
from utils import make_serise, make_serise_clip

from .restoration import RRDB_Block

#from blind_model_2_1.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
class PE_head(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.head = nn.Sequential(*[
            nn.Conv2d(1, nf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        ])
        self.body = nn.Sequential(*[RRDB_Block(nf=nf,gc=nf//4) for _ in range(1)])
        self.tail = nn.Sequential(*[
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
        ])  
    def forward(self, x):
        residual = self.head(x)
        x = self.body(residual) + residual
        x = self.tail(x)
        return x