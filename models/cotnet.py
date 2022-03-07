import math
import numpy as np
import torch
from torch import nn as nn

from config import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import SelectiveKernelConv, ConvBnAct, create_attn
from .registry import register_model
from .resnet import ResNet
from .layers import Shiftlution
from cupy_layers.aggregation_zeropad import LocalConvolution
from .layers import create_act_layer
import torch.nn.functional as F
from torch import einsum
from .layers.activations import Swish
from .layers.tbconv import TBConv
from .layers.kerv2d import Kerv2d
from .layers import get_act_layer

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

default_cfgs = {
    'cot_basic': _cfg(
        url=''),
}

class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2

        # 两个1x1卷积，θ和δ
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        # 用于qkv中的v
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        # 激活函数 active
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)

        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        # 获取static Contexts：CBL三元组 （3×3）
        k = self.key_embed(x)
        # 将查询特征 q 和键 k 拼接起来（Concat）
        qk = torch.cat([x, k], dim=1)
        # 获取qk矩阵的 B, C, H, W
        b, c, qk_hh, qk_ww = qk.size()

        # 两个1×1卷积层：CBL + CG
        # 1. 第二个卷积层没有ReLU，可以避免屏蔽过多的有效信息
        # 2. 第二个卷积层使用GroupNorm，将channel分为G组，每组分别做归一化，避免不同batch特征之间造成干扰
        w = self.embed(qk)
        # 展开为多个头部
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_hh, qk_ww)

        # 获取 value（qkv中的v）
        # Conv 1×1 + BN
        x = self.conv1x1(x)
        # 获取dynamic contexts：结合权重 w 和 值 v 作矩阵乘法，对每个空间像素进行加权
        x = self.local_conv(x, w)
        # Conv 3×3 + BN + swish
        x = self.bn(x)
        x = self.act(x)

        # 增强维度用于特征融合
        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        # 将 dynamic contexts 和 static contexts 拼接起来（Concat）
        x = torch.cat([x, k], dim=2)

        # 融合 dynamic contexts 和 static contexts，并基于此再进行注意力加权
        x_gap = x.sum(dim=2)
        # 对特征图的空间维度求均值
        x_gap = x_gap.mean((2, 3), keepdim=True)
        # CBLC （1×1）特征编码
        x_attn = self.se(x_gap)
        # 特征维度变换
        x_attn = x_attn.view(B, C, self.radix)
        # 计算特征重要性
        x_attn = F.softmax(x_attn, dim=2)
        # 将注意力图重新加权回 x
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
        # 使tensor在内存中连续存放
        return out.contiguous()

class CoXtLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CoXtLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=8, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.dw_group = 2
        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, groups=self.dw_group, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1, groups=self.dw_group),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.dw_group, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        k = self.key_embed(x)
        qk = torch.cat([x.unsqueeze(2), k.unsqueeze(2)], dim=2)
        qk = qk.view(batch_size, -1, height, width)

        w = self.embed(qk)
        w = w.view(batch_size * self.dw_group, 1, -1, self.kernel_size*self.kernel_size, height, width)
        
        x = self.conv1x1(x)
        x = x.view(batch_size * self.dw_group, -1, height, width)
        x = self.local_conv(x, w)
        x = x.view(batch_size, -1, height, width)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
        
        return out.contiguous()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        if stride > 1:
            self.avd = nn.AvgPool2d(3, 2, padding=1)
        else:
            self.avd = None
        
        self.conv2 = CotLayer(width, kernel_size=3) if cardinality == 1 else CoXtLayer(width, kernel_size=3)

        #self.conv2 = nn.Conv2d(
        #    first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #    padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        #self.bn2 = norm_layer(width)
        #self.act2 = act_layer(inplace=True)
        #self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        if self.avd is not None:
            x = self.avd(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        #if self.drop_block is not None:
        #    x = self.drop_block(x)
        #x = self.act2(x)
        #if self.aa is not None:
        #    x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x

def _create_cotnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)

@register_model
def cotnet50(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_cotnet('cot_basic', pretrained, **model_args)

@register_model
def cotnext50_2x48d(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=2, base_width=48, **kwargs)
    return _create_cotnet('cot_basic', pretrained, **model_args)

@register_model
def cotnet101(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_cotnet('cot_basic', pretrained, **model_args)

@register_model
def cotnext101_2x48d(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=2, base_width=48, **kwargs)
    return _create_cotnet('cot_basic', pretrained, **model_args)
