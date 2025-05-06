import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os
import math
import numpy as np
from torch import Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys

from sk_att_fusion import SK_channel_my

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out,group=1, k=1, p=0,s=1, bias=False,init_weight=True):
        super(single_conv, self).__init__()
        self.s_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=k, stride=s, padding=p, bias=bias,groups=group),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU()
            # nn.GELU()
            nn.SiLU()
        )
        if init_weight:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.s_conv(x)
        return x
class double_conv(nn.Module):
    def __init__(self, ch_in, ch_out,k=3,p=1,s=1,init_weight=True):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            single_conv(ch_in, ch_out,k=k, p=p,s=s),
            single_conv(ch_out, ch_out,k=k, p=p,s=s)
        )

        if init_weight:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.d_conv(x)
        # if self.ds:
        #     x = self.downsample(x) #128
        return x
class up(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, init_weight=True):
        super(up, self).__init__()
        self.up_2d = nn.Sequential(
            nn.Upsample(scale_factor=2),
            single_conv(ch_in, ch_out, k=kernel_size, p=padding, s=stride, bias=False,init_weight=True)
        )
        if init_weight:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.up_2d(x)
        return x
class Attention_block(nn.Module):  # attention Gate
    def __init__(self, F_l,F_o):
        super(Attention_block, self).__init__()

        self.SK_channel_my = SK_channel_my(F_l,F_o)

    def forward(self, x0, x1):

        fusion = torch.cat((x0, x1),dim=1)
        se = self.SK_channel_my(fusion)

        return se

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int,
                 group_num:int = 16,
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.contiguous().view(N, self.group_num, -1)
        mean = x.mean(dim = 2, keepdim = True)
        std = x.std (dim = 2, keepdim = True)
        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int,
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels = oup_channels, num_groups = group_num) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self,x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight/torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1,-1,1,1)
        reweigts = self.sigomid( gn_x * w_gamma )
        # Gate
        info_mask = reweigts>=self.gate_treshold
        noninfo_mask= reweigts<self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1,x_2)
        return x

    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)
class CRU(nn.Module):
    '''
    gamma: 0<gamma<1
    '''
    def __init__(self,
                 op_channel:int,
                 gamma:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(gamma*op_channel)
        self.low_channel = low_channel = op_channel-up_channel
        self.squeeze1 = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2 = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1 = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2 = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out = torch.cat( [Y1,Y2], dim= 1 )
        out = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2
class ScConv(nn.Module):
    def __init__(self,
                 op_channel:int,
                 group_num:int = 4,
                 gate_treshold:float = 0.5,
                 gamma:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                        group_num = group_num,
                        gate_treshold = gate_treshold)
        self.CRU = CRU( op_channel,
                        gamma = gamma,
                        squeeze_radio = squeeze_radio,
                        group_size = group_size,
                        group_kernel_size = group_kernel_size)
    def forward(self,x):
        x1 = self.SRU(x)
        y = self.CRU(x1)
        return y