import math
from torch import Tensor
from einops import rearrange
import torch
import torch.nn as nn
import sys
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
import torch.nn.functional as F
from functools import partial
from bunch import Bunch
from yaml import safe_load
from torch.utils import tensorboard
from ptflops import get_model_complexity_info

from module import double_conv, up, Attention_block
from block import Mlp, initAttention, initBlock, Attention, Block, OverlapPatchEmbed, Head, Bottleneck
from dcn import DeformConv2d

class DC(nn.Module):

    def __init__(self, c_i, c_o=None):
        super().__init__()
        if c_o == None:
            c_o = c_i

        self.dc = nn.Sequential(DeformConv2d(c_i, c_o, modulation=True))
                                # nn.BatchNorm2d(c_o), nn.SiLU())
        # nn.Conv2d(c_o, 1, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        dc = self.dc(x)
        return dc

class SAM2T(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, im_size=48, embed_dims=[32, 64, 128, 256, 512],
                 ca_num_heads=[4, 4, -1], sa_num_heads=[-1, 8, 16], mlp_ratios=[2, 2, 1],
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 Endepths=[2, 8, 1], ca_attentions=[0, 1, 0], num_stages=4, head_conv=3, expand_ratio=2, **kwargs):
        super(SAM2T, self).__init__()

        # En
        # stem
        self.dc0 = DC(c_i=img_ch,c_o=embed_dims[0])

        # stage0
        self.patch_embed0 = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=embed_dims[0], embed_dim=embed_dims[0]) 
        self.block0 = nn.ModuleList([initBlock(
            dim=embed_dims[0], ca_num_heads=ca_num_heads[0], sa_num_heads=sa_num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.1, norm_layer=norm_layer,
            ca_attention=0 if j%2!=0 else ca_attentions[0], expand_ratio=expand_ratio) 
            for j in range(Endepths[0])])

        # stage1
        self.patch_embed1 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]) 
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[1], ca_num_heads=ca_num_heads[1], sa_num_heads=sa_num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.1, norm_layer=norm_layer,
            ca_attention=0 if j%2!=0 else ca_attentions[1], expand_ratio=expand_ratio, local=True) 
            for j in range(Endepths[1])])

        # top
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]) 
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[2], ca_num_heads=ca_num_heads[2], sa_num_heads=sa_num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.1, norm_layer=norm_layer,
            ca_attention=ca_attentions[2], expand_ratio=expand_ratio, local=False) # 0 1 2 3
            for j in range(Endepths[2])])

        # skip
        self.ds2 = nn.MaxPool2d(2,2)
        self.us2 = nn.Upsample(scale_factor=2)

        # De
        # de stage1
        self.Up2 = up(ch_in=embed_dims[2], ch_out=embed_dims[1]) # 1 layers
        self.Att2 = Attention_block(F_l=embed_dims[0]+embed_dims[1], F_o=embed_dims[1])
        self.Up_conv2 = double_conv(ch_in=embed_dims[2], ch_out=embed_dims[1])

        # de stage0
        self.Up1 = up(ch_in=embed_dims[1], ch_out=embed_dims[0]) # 1 layers
        self.Att1 = Attention_block(F_l=embed_dims[0]+embed_dims[1], F_o=embed_dims[0])
        self.Up_conv1 = double_conv(ch_in=embed_dims[1], ch_out=embed_dims[0])

        # head
        self.Conv_1x1 = nn.Conv2d(embed_dims[0], output_ch, kernel_size=1, stride=1, padding=0) # 1 layers

    def forward(self, x):
        B = x.shape[0]
        x = self.dc0(x)
        x0, H, W = self.patch_embed0(x)
        for blk in self.block0:
            x0 = blk(x0, H, W)
        C = x0.shape[2]
        x0 = x0.contiguous().view(B, H, W, C).permute(0, 3, 1, 2)
        sk0 = x0

        x1, H, W = self.patch_embed1(x0)
        for blk in self.block1:
            x1 = blk(x1, H, W)
        C = x1.shape[2]
        x1 = x1.contiguous().view(B, H, W, C).permute(0, 3, 1, 2)
        sk1 = x1

        x2, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2, H, W)
        C = x2.shape[2]
        x2 = x2.contiguous().view(B, H, W, C).permute(0, 3, 1, 2)

        skip1 = self.Att2(self.ds2(sk0), sk1)
        skip0 = self.Att1(sk0, self.us2(sk1))

        d2 = self.Up2(x2)
        d2 = torch.cat((skip1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((skip0, d1), dim=1)
        d1 = self.Up_conv1(d1)

        output = self.Conv_1x1(d1)

        return output

if __name__ == '__main__':

    H = W = 64
    batch_size = 1
    C = 1
    x = torch.randn((batch_size, 1, H, W)).cuda()
    model = SAM2T(img_size=H).cuda()
    out=model(x)
    print(out.shape)
    ### ptflops cal ###
    print("**********ptflops cal**********")
    for i in range(1):
        flops_count, params_count = get_model_complexity_info(model,(C, H, W), as_strings=True, print_per_layer_stat=False)
        print('flops: ', flops_count)
        print('params: ', params_count)



        
    



