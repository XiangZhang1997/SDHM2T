import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from module import ScConv

class basic_f(nn.Module):
    def __init__(self,embedding_dim,head_num=4): #512 4
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.HighMixer = ScConv(self.embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self,x, H, W):
        x = x.permute(0,2,1,3).flatten(2) # torch.Size([B, 4, 4096, 128])--torch.Size([B, 4096, 4, 128]) ---torch.Size([B, 4096, 128*4]) for LN
        x = self.norm(x)
        B,num_patches,emb_dim = x.shape # torch.Size([B, 4096, 128*4]) where num_patches=4096
        img_size = int(num_patches **0.5) # 64 or 80 获取img_size
        x = x.reshape(B,-1,img_size,img_size) # torch.Size([B, 32, 64, 64])
        x = self.HighMixer(x)
        y = x.reshape(B,num_patches,self.head_num,self.embedding_dim//self.head_num).permute(0,2,1,3) # torch.Size([B, 4096, 4, 128])
        return y
class initAttention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2, local=False):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads
        self.local = local
        self.split_groups=self.dim//ca_num_heads
        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."
        # self.act = nn.GELU()
        # self.act = nn.ReLU()
        self.act = nn.SiLU()
        # self.sc = ScConv(self.dim)
        self.proj0 = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=dim) # 分组（4）卷积 C--2C
        self.bn = nn.BatchNorm2d(dim*expand_ratio)
        self.proj1 = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1) # 2C--C
        self.proj = nn.Linear(dim, dim,bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        if ca_attention == 1:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            # self.q = nn.Conv2d(dim,dim,kernel_size=1,stride=1)
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads, bias=False)
                setattr(self, f"local_conv_{i + 1}", local_conv)

        else:
            self.ca_attn_drop = nn.Dropout(attn_drop)
            ca_head_dim = dim // ca_num_heads  # 8 16 group
            self.ca_scale = (1+1e-6) / (math.sqrt(ca_head_dim)+1e-6)  # 0.35355339059327373  0.25
            # self.pool_avg = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
            # self.pool_max = nn.MaxPool2d(3, stride=1, padding=1, count_include_pad=False)
            self.pool_n1 = nn.AdaptiveAvgPool2d((None,1))  # (B,C,1,W)
            self.pool_1n = nn.AdaptiveAvgPool2d((1,None))  # (B,C,1,W)

            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)

            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads)
                local_conv_x1 = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2,1), padding=(1+i,1), stride=1, groups=dim//self.ca_num_heads)
                local_conv_1x = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(1,3+i*2), padding=(1,1+i), stride=1, groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
                setattr(self, f"local_conv_x1_{i + 1}", local_conv_x1)
                setattr(self, f"local_conv_1x_{i + 1}", local_conv_1x)
            self.conv_ = nn.Conv2d(dim*3, dim, kernel_size=1, padding=0, stride=1,bias=False)
            # self.conv_ = ScConv(dim*2,dim)
            self.bn_ = nn.BatchNorm2d(dim)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def convF(self,s,B,N,C,H,W):
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                local_conv_x1 = getattr(self, f"local_conv_x1_{i + 1}")
                local_conv_1x = getattr(self, f"local_conv_1x_{i + 1}")

                #x
                s_i= s[i] # (1, B, C/h, H, W)
                # s_i = self.pool_avg(s_i) + self.pool_max(s_i)
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out,s_i],2) # (1, B, C/h, H, W) cat (1, B, C/h, H, W)

                #x1
                s_i_x1= s[i] # (1, B, C/h, H, W)
                s_i_x1 = self.pool_1n(s_i_x1)
                s_i_x1 = F.interpolate(local_conv_x1(s_i_x1),(H, W)).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out_x1 = s_i_x1
                else:
                    s_out_x1 = torch.cat([s_out_x1,s_i_x1],2) # (1, B, C/h, H, W) cat (1, B, C/h, H, W)

                #1x
                s_i_1x= s[i] # (1, B, C/h, H, W)
                s_i_1x = self.pool_n1(s_i_1x)
                s_i_1x = F.interpolate(local_conv_1x(s_i_1x),(H, W)).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out_1x = s_i_1x
                else:
                    s_out_1x = torch.cat([s_out_1x,s_i_1x],2) # (1, B, C/h, H, W) cat (1, B, C/h, H, W)

            s_out = s_out.reshape(B, C, H, W) # (1, B, C, H, W)--(B, C, H, W)
            s_out_x1 = s_out_x1.reshape(B, C, H, W) # (1, B, C, H, W)--(B, C, H, W)
            s_out_1x = s_out_1x.reshape(B, C, H, W) # (1, B, C, H, W)--(B, C, H, W)

            # Scale-Aware Aggregation
            s_out_f = self.conv_(torch.cat((s_out,s_out_x1,s_out_1x),dim=1))
            s_out_f = s_out_f.reshape(B, C, N).permute(0, 2, 1) # (B, C, H, W)--(B, C, H*W)--(B, H*W, C)
            x = s_out_f
            return x

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.ca_attention == 1:
            q = self.q(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2) # (h, B, C/h, H, W) h==4
            # multi-scale Conv self-attn
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i= q[i] # (1, B, C/h, H, W)
                s_i = local_conv(s_i)
                s_i = s_i.reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out,s_i],2) # (1, B, C/h, H, W) cat (1, B, C/h, H, W)
            s_out = s_out.reshape(B, C, H, W) # (1, B, C, H, W)--(B, C, H, W)
            # s_out = self.sc(s_out)
            x = self.proj1(self.act(self.bn(self.proj0(s_out))))
            x = x.reshape(B, C, N).permute(0, 2, 1) # (B, C, H, W)--(B, C, H*W)--(B, H*W, C)
        else:
            q = self.q(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2) # (h, B, C/h, H, W) h==4
            k = self.k(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2) # (h, B, C/h, H, W) h==4
            v = self.v(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2) # (h, B, C/h, H, W) h==4
            # multi-scale Conv self-attn
            q_cf = self.convF(q,B,N,C,H,W)
            k_cf = self.convF(k,B,N,C,H,W)
            v_cf = self.convF(v,B,N,C,H,W)
            # Conv self-attn Ehanced here!!!!!!!!!!!
            attn = (q_cf * k_cf)*self.ca_scale
            attn = self.ca_attn_drop(attn)
            x = (attn * v_cf).reshape(B, C, N).permute(0, 2, 1) # (B, C, H, W)--(B, C, H*W)--(B, H*W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class initBlock(nn.Module):

    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1,expand_ratio=2, local=False):
        super().__init__()
        self.ca_attention = ca_attention
        self.norm1 = norm_layer(dim)
        self.attn = initAttention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=self.ca_attention,
            expand_ratio=expand_ratio, local=local)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        if self.ca_attention==1:
            x = self.attn(x, H, W)
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))

        return x
class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2, local=False):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads
        self.local = local
        self.split_groups=self.dim//ca_num_heads
        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."
        # self.act = nn.GELU()
        # self.act = nn.ReLU()
        self.act = nn.SiLU()
        self.proj0 = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=dim) # 分组（4）卷积 C--2C
        self.bn = nn.BatchNorm2d(dim*expand_ratio)
        self.proj1 = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1) # 2C--C
        self.proj = nn.Linear(dim, dim,bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        if ca_attention == 1:
            self.ca_attn_drop = nn.Dropout(attn_drop)
            ca_head_dim = dim // ca_num_heads  # 8 16
            self.ca_scale = (1+1e-6) / (math.sqrt(ca_head_dim)+1e-6)  # 0.35355339059327373  0.25
            # self.pool_avg = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
            self.pool_n1 = nn.AdaptiveAvgPool2d((None,1))  # (B,C,1,W)
            self.pool_1n = nn.AdaptiveAvgPool2d((1,None))  # (B,C,1,W)

            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)

            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads)
                local_conv_x1 = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2,1), padding=(1+i,1), stride=1, groups=dim//self.ca_num_heads)
                local_conv_1x = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(1,3+i*2), padding=(1,1+i), stride=1, groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
                setattr(self, f"local_conv_x1_{i + 1}", local_conv_x1)
                setattr(self, f"local_conv_1x_{i + 1}", local_conv_1x)
            self.conv_ = nn.Conv2d(dim*3, dim, kernel_size=1, padding=0, stride=1,bias=True)
            self.bn_ = nn.BatchNorm2d(dim)

        # SA
        else:
            self.dwconv = DWConv(self.dim,3,1,1)
            self.maxpool = nn.MaxPool2d(3,1,1)
            self.enhanced_module = basic_f(self.dim,head_num=self.sa_num_heads) # embedding_dim is channel_num
            head_dim = dim // sa_num_heads   # group
            self.scale = qk_scale or (1+1e-6) / (math.sqrt(head_dim)+1e-6) # qk_scale=None
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def convF(self,s,B,N,C,H,W):
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            local_conv_x1 = getattr(self, f"local_conv_x1_{i + 1}")
            local_conv_1x = getattr(self, f"local_conv_1x_{i + 1}")

            #x
            s_i= s[i] # (1, B, C/h, H, W)
            # s_i = self.pool_avg(s_i) + self.pool_max(s_i)
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out,s_i],2) # (1, B, C/h, H, W) cat (1, B, C/h, H, W)

            #x1
            s_i_x1= s[i] # (1, B, C/h, H, W)
            s_i_x1 = self.pool_1n(s_i_x1)
            s_i_x1 = F.interpolate(local_conv_x1(s_i_x1),(H, W)).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out_x1 = s_i_x1
            else:
                s_out_x1 = torch.cat([s_out_x1,s_i_x1],2) # (1, B, C/h, H, W) cat (1, B, C/h, H, W)

            #1x
            s_i_1x= s[i] # (1, B, C/h, H, W)
            s_i_1x = self.pool_n1(s_i_1x)
            s_i_1x = F.interpolate(local_conv_1x(s_i_1x),(H, W)).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out_1x = s_i_1x
            else:
                s_out_1x = torch.cat([s_out_1x,s_i_1x],2) # (1, B, C/h, H, W) cat (1, B, C/h, H, W)

        s_out = s_out.reshape(B, C, H, W) # (1, B, C, H, W)--(B, C, H, W)
        s_out_x1 = s_out_x1.reshape(B, C, H, W) # (1, B, C, H, W)--(B, C, H, W)
        s_out_1x = s_out_1x.reshape(B, C, H, W) # (1, B, C, H, W)--(B, C, H, W)

        # Scale-Aware Aggregation
        s_out_f = self.conv_(torch.cat((s_out,s_out_x1,s_out_1x),dim=1))
        s_out_f = s_out_f.reshape(B, C, N).permute(0, 2, 1) # (B, C, H, W)--(B, C, H*W)--(B, H*W, C)
        x = s_out_f
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape
        # ConvFormer with kernel
        if self.ca_attention == 1:
            # x1 = self.dwconv(x,H,W)
            # x1 = x1.flatten(2).transpose(1, 2)
            q = self.q(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2) # (h, B, C/h, H, W) h==4
            k = self.k(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2) # (h, B, C/h, H, W) h==4
            v = self.v(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2) # (h, B, C/h, H, W) h==4
            # multi-scale Conv self-attn
            q_cf = self.convF(q,B,N,C,H,W)
            k_cf = self.convF(k,B,N,C,H,W)
            v_cf = self.convF(v,B,N,C,H,W)
            # Conv self-attn Ehanced here!!!!!!!!!!!
            attn = (q_cf * k_cf)*self.ca_scale
            attn = self.ca_attn_drop(attn)
            x = (attn * v_cf).reshape(B, C, N).permute(0, 2, 1) # (B, C, H, W)--(B, C, H*W)--(B, H*W, C)
        else:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3) # (B, p^2, head, C//head)-->(B, head, P^2, C//head)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4) # (B, p^2, 2, head, C//head)-->(2, B, head, P^2, C//head)
            k, v = kv[0], kv[1]

            if self.local:
                # channel& sptial enhanced qkv
                q = self.enhanced_module(q, H, W)
                k = self.enhanced_module(k, H, W)
                v = self.enhanced_module(v, H, W)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W).flatten(2).transpose(1, 2))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Block(nn.Module):

    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                    use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1,expand_ratio=2, local=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention, 
            expand_ratio=expand_ratio, local=local)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if ca_attention == 1:
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))

        return x
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=3, stride=2, in_chans=1, embed_dim=16):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2),bias=False) #
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x) # map x into a sequence x1, (GT-DLA)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # x: (B, C, H*W)--(B, H*W, C)
        x = self.norm(x)
        return x, H, W
class Head(nn.Module):
    def __init__(self, head_conv, dim): # 3,64
        super(Head, self).__init__()
        stem = [nn.Conv2d(1, dim, head_conv, 2, padding=3 if head_conv==7 else 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True)]
        stem.append(nn.Conv2d(dim, dim, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem) # double conv Size--Size/4
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv(x)
        _, _, H, W = x.shape # B,C,H/4,W/4
        x = x.flatten(2).transpose(1, 2) # B,C,H*W/16--# B,H*W/16,C
        x = self.norm(x)
        return x, H, W
class DWConv(nn.Module):
    def __init__(self, dim=768,k=3,s=1,p=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=True, groups=dim)

    def forward(self, x, H=None, W=None):
        # print(x.shape)
        if len(x.shape) == 3:
            B, N, C = x.shape
            x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)

        return x