# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path
from einops import rearrange

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized
import torch.nn.functional as F


# def autopad(k, p=None):  # kernel, padding  # 自动padding,不指定p时自动按kernel大小pading到same
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad  # k为可迭代对象时,支持同时计算多个pading量
#     return p

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution  # 分组卷积, 组数取c1, c2(输入输出通道)最大公因数
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution  通用卷积模块,包括1卷积1BN1激活,激活默认SiLU,可用变量指定,不激活时用nn.Identity()占位,直接返回输入
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck 残差块
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):  # 如果shortcut并且输入输出通道相同则跳层相加
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):  # 5.0模型没用这个, 和C3区别在于 C3 cat后一个卷积,这个cat后BN激活再卷积
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):  # 5.0版本模型backbone和head用的都是这个
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])  # n个残差组件(Bottleneck)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3SPP(nn.Module):  
    def __init__(self, c1, c2, k=(5, 9, 13), g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3SPP, self).__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_ + int(c_*1.5), c2, 1)  # act=FReLU(c2)
        self.m = SPP(c_, int(c_*1.5), k=k)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP # ModuleLis容器多分支实现SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 输入卷一次
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)  # 输出卷一次(输入通道:SPP的len(k)个尺度cat后加输入)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Attention(nn.Module):
    def __init__(self, chan, reduction=1):
        super(Attention, self).__init__()
        if reduction > 1:
            self.W =  nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                    Conv(chan, chan//reduction, k=1, s=1), 
                                    Conv(chan//reduction, chan, k=1, s=1, act=False),                                
                                    nn.Sigmoid()
                                    )
        else:
            self.W =  nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                    Conv(chan, chan, k=1, s=1, act=False),                                
                                    nn.Sigmoid()
                                    )
    def forward(self, x):
        return x * self.W(x)


class ARM(nn.Module):   # AttentionRefinementModule
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(ARM, self).__init__()
        self.conv = Conv(in_chan, out_chan, k=3, s=1, p=None)  #　Conv 自动padding
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),  # ARM的SE带bn不带act
                                               Conv(out_chan, out_chan, k=1, s=1,act=False),   # 注意ARM的SE处用了BN，FFM没用，SE用了BN的模型training时不支持单个样本，对应改了两处，一是yolo.py构造好跑一次改成了(2,3,256,256)
                                               nn.Sigmoid()                 # 二是train.py的batch开头加了一句单样本时候continue(分割loader容易加droplast，但是检测loader出现地方太多没分mode不好改)
                                               )            

    def forward(self, x):
        feat = self.conv(x)  # 先3*3卷积一次
        atten = self.channel_attention(feat)  # SE
        return torch.mul(feat, atten)
            

class FFM(nn.Module):  # FeatureFusionModule  reduction用来控制瓶颈结构
    def __init__(self, in_chan, out_chan, reduction=1, is_cat=True, k=1):
        super(FFM, self).__init__()
        self.convblk = Conv(in_chan, out_chan, k=k, s=1, p=None)  ## 注意力处用了１＊１瓶颈，两个卷积都不带bn,一个带普通激活，一个sigmoid
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(out_chan, out_chan//reduction,
                                                         kernel_size = 1, stride = 1, padding = 0, bias = False),
                                               nn.SiLU(inplace=True),
                                               nn.Conv2d(out_chan//reduction, out_chan,
                                                         kernel_size = 1, stride = 1, padding = 0, bias = False),
                                               nn.Sigmoid(),
                                            )
        self.is_cat = is_cat

    def forward(self, fspfcp):  #空间, 语义两个张量用[]包裹送入模块，为了方便Sequential
        fcat = torch.cat(fspfcp, dim=1) if self.is_cat else fspfcp
        feat = self.convblk(fcat)
        atten = self.channel_attention(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class ASPP(nn.Module):  # ASPP，原版没有hid，为了灵活性方便砍通道增加hid，hid和out一样就是原版
    def __init__(self, in_planes, out_planes, d=[3, 6, 9], has_globel=True, map_reduce=4):
        super(ASPP, self).__init__()
        self.has_globel = has_globel
        self.hid = in_planes//map_reduce

        self.branch0 = nn.Sequential(
                Conv(in_planes, self.hid, k=1, s=1),
                )
        self.branch1 = nn.Sequential(
                nn.Conv2d(in_planes, self.hid, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()    
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(in_planes, self.hid, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()                    
                )
        self.branch3 = nn.Sequential(
                nn.Conv2d(in_planes, self.hid, kernel_size=3, stride=1, padding=d[2], dilation=d[2], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()    
                )
        if self.has_globel:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(in_planes, self.hid, k=1),
                )
        self.ConvLinear = Conv(int(5*self.hid) if has_globel else int(4*self.hid), out_planes, k=1, s=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if not self.has_globel:
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3],1))
            return out
        else:
            x4 = F.interpolate(self.branch4(x), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3,x4],1))
            return out


class ASPPs(nn.Module):  # 空洞卷积前先用1*1砍通道到目标（即相比上面版本空洞卷积的输入通道减少，一个1*1统一砍通道试过效果不好，每个分支1*1独立,1*1分支改3*3）
    def __init__(self, in_planes, out_planes, d=[3, 6, 9], has_globel=True, map_reduce=4):
        super(ASPPs, self).__init__()
        self.has_globel = has_globel
        self.hid = in_planes//map_reduce

        self.branch0 = nn.Sequential(
                Conv(in_planes, self.hid, k=1),
                Conv(self.hid, self.hid, k=3, s=1),
                )
        self.branch1 = nn.Sequential(
                Conv(in_planes, self.hid, k=1),
                nn.Conv2d(self.hid, self.hid, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()    
                )
        self.branch2 = nn.Sequential(
                Conv(in_planes, self.hid, k=1),    
                nn.Conv2d(self.hid, self.hid, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()                    
                )
        self.branch3 = nn.Sequential(
                Conv(in_planes, self.hid, k=1),
                nn.Conv2d(self.hid, self.hid, kernel_size=3, stride=1, padding=d[2], dilation=d[2], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()    
                )
        if self.has_globel:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(in_planes, self.hid, k=1),
                )
        self.ConvLinear = Conv(int(5*self.hid) if has_globel else int(4*self.hid), out_planes, k=1, s=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if not self.has_globel:
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3],1))
            return out
        else:
            x4 = F.interpolate(self.branch4(x), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3,x4],1))
            return out


class DAPPM(nn.Module):
    """
    https://github.com/ydhongHIT/DDRNet，只换了激活函数，原仓库代码每个Block里Conv,BN,Activation的顺序写法很非主流,这种非主流写法应该也是考虑了两个层相加后再进行BN和激活
    使用注意，若遵照原作者用法，1、此模块前一个Block只Conv，不BN和激活（因为每个scale pooling后BN和激活）；
                           2、此模块后一个Block先BN和激活再接其他卷积层（模块结束后与高分辨率相加后统一BN和激活，与之相加的高分辨率的上一Block最后也不带BN和激活）
    """
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes * 5),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 


# 和ASPPs类似(初衷都是为了砍ASPP计算量，这个模块砍中间和输入通道增加3*3卷积补偿;ASPPs砍中间和输入通道，没有多的操作，同延时下可以少砍一点)
class RFB1(nn.Module):  # 魔改ASPP和RFB,这个模块其实长得更像ASPP,相比RFB少shortcut,３＊３没有宽高分离,d没有按照RFB设置;相比ASPP多了1*1砍输入通道和3*3卷积
    def __init__(self, in_planes, out_planes, map_reduce=4, d=[3, 5, 7], has_globel=False):
        super(RFB1, self).__init__()
        self.out_channels = out_planes
        self.has_globel = has_globel
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=3, s=1)
                )
        self.branch1 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=3, s=1),
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()    
                )
        self.branch2 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=3, s=1),
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()                    
                )
        self.branch3 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=5, s=1),
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[2], dilation=d[2], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()    
                )
        if self.has_globel:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(in_planes, inter_planes, k=1),
                )
        self.Fusion = Conv(int(5*inter_planes) if has_globel else int(4*inter_planes), out_planes, k=1, s=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if not self.has_globel:
            out = self.Fusion(torch.cat([x0,x1,x2,x3], 1))
            return out
        else:
            x4 = F.interpolate(self.branch4(x), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.Fusion(torch.cat([x0,x1,x2,x3,x4],1))
            return out



class RFB2(nn.Module):  # 魔改模块,除了历史遗留(改完训练模型精度不错，不想改名重训)名字叫RFB，其实和RFB没啥关系了(参考deeplabv3的反面级联结构，也有点像CSP，由于是级联，d设置参考论文HDC避免网格效应)实验效果不错，能满足较好非线性、扩大感受野、多尺度融合的初衷(在bise中单个精度和多个其他模块组合差不多，速度和C3相近比ASPP之类的快)
    def __init__(self, in_planes, out_planes, map_reduce=4, d=[2, 3], has_globel=False):  # 第一个3*3的d相当于1，典型的设置1,2,3; 1,2,5; 1,3,5
        super(RFB2, self).__init__()
        self.out_channels = out_planes
        self.has_globel = has_globel
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=3, s=1)
                )
        self.branch1 = nn.Sequential(
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()    
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()                    
                )
        self.branch3 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),  
                )
        if self.has_globel:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(inter_planes, inter_planes, k=1),
                )
        self.ConvLinear = Conv(int(5*inter_planes) if has_globel else int(4*inter_planes), out_planes, k=1, s=1)

    def forward(self, x):  # 思路就是rate逐渐递进的空洞卷积连续卷扩大感受野避免使用rate太大的卷积(级联注意rate要满足HDC公式且不应该有非1公倍数，空洞卷积网格效应)，多个并联获取多尺度特征
        x3 = self.branch3(x)  # １＊１是独立的　类似C3，区别在于全部都会cat
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        if not self.has_globel:
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3],1))
        else:
            x4 = F.interpolate(self.branch4(x2), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3,x4],1))
        return out


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, k=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(k[0])
        self.pool2 = nn.AdaptiveAvgPool2d(k[1])
        self.pool3 = nn.AdaptiveAvgPool2d(k[2])
        self.pool4 = nn.AdaptiveAvgPool2d(k[3])

        out_channels = in_channels//4
        self.conv1 = Conv(in_channels, out_channels, k=1)
        self.conv2 = Conv(in_channels, out_channels, k=1)
        self.conv3 = Conv(in_channels, out_channels, k=1)
        self.conv4 = Conv(in_channels, out_channels, k=1)

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode='bilinear', align_corners=True)

        return torch.cat((x, feat1, feat2, feat3, feat4, ), 1)


class Focus(nn.Module):  # 卷积复杂度O(W*H*C_in*C_out)此操作使WH减半,后续C_in翻4倍, 把宽高信息整合到通道维度上,
    # Focus wh information into c-space  # 相同下采样条件下计算量会减小,　后面Contract, Expand用不同的方法实现同样的目的
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):  # 用nn.Module包装了cat方法
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):  # 用nn.Module包装了nms函数
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:  # 类用于模型inference结束后输入输出的后处理(赋予类名,打印,显示,保存)
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = self.files[i]
                img.save(Path(save_dir) / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n



#分类新加models

import cv2
import json
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_notebook, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from urllib.parse import urlparse

class DetectMultiBackend(nn.Module):
    # YOLO MultiBackend class for python inference on various backends
    def __init__(self, weights='yolo.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load1  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load1(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, im2, augment=False, visualize=False):
        # YOLO MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
            im2 = im2.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im2 = im2.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, im2, augment=augment, visualize=visualize) if augment or visualize else self.model(im, im2)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im, im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from models.export import export_formats, is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None




class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

#新版yolo模块
class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# RGB-Depth 特征融合模块
class SE(nn.Module):
    def __init__(self, c1, reduction=16, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, kernel_size=1),
            activation,
            nn.Conv2d(c1 // reduction, c1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        # return weighting
        y = x * weighting
        return y

# 最后不add直接conv
class SEFusion(nn.Module):
    def __init__(self, c1, activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.se_rgb = SE(c1, activation=activation)
        self.se_depth = SE(c1, activation=activation)
        self.cv1 = Conv(2*c1, c1, 1, 1)

    def forward(self, x):
        rgb = self.se_rgb(x[1])
        depth = self.se_depth(x[0])
        out = self.cv1(torch.cat((rgb, depth), dim=1))
        return out

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2),
                    nn.Sigmoid())

    def forward(self, x):
        B, _, H, W = x[0].shape
        x = torch.cat((x[0], x[1]), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights

class FRM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.cv1 = Conv(2*dim, dim, 1, 1)

    def forward(self, x):
        channel_weights = self.channel_weights(x)
        out_x1 = x[0] + self.lambda_c * channel_weights[1] * x[1]
        out_x2 = x[1] + self.lambda_c * channel_weights[0] * x[0]
        out = self.cv1(torch.cat((out_x1, out_x2), dim=1))
        # return out_x1 + out_x2
        return out
#CA注意力
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class CAWeights(nn.Module):
    def __init__(self, c1, c2, reduction=32):
        super().__init__()
        self.c1 = c1

        self.avg_h = nn.AdaptiveAvgPool2d((None, 1))
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.avg_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_w = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, 4 * c1 // reduction)

        # self.conv1 = nn.Conv2d(4 * c1, mip, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(2 * c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, 2 * c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, 2 * c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.cat((x[0], x[1]), dim=1)

        b, c, h, w = x.size()                      #n*2c*h*w
        x_ah = self.avg_h(x)                       #n*2c*h*1
        x_mh = self.max_h(x)
        x_aw = self.avg_w(x).permute(0, 1, 3, 2)   #n*2c*1*w
        x_mw = self.max_w(x).permute(0, 1, 3, 2)

        y_a = torch.cat([x_ah, x_aw], dim=2)       #n*2c*h+w*1
        y_m = torch.cat([x_mh, x_mw], dim=2)
        # y = torch.cat((y_a, y_m), dim=1)           #n*4c*h+w*1
        y = torch.cat((x_mh, x_mw), dim=2)
        # y = torch.cat((x_ah, x_aw), dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)   #n*4c*h/w*1
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()           #n*2c*h*1
        a_w = self.conv_w(x_w).sigmoid()           #n*2c*1*w

        h_weights = a_h.reshape(b, 2, self.c1, h, 1).permute(1, 0, 2, 3, 4)
        w_weights = a_w.reshape(b, 2, self.c1, 1, w).permute(1, 0, 2, 3, 4)

        return h_weights, w_weights

#4\6 层融合时使用的CAFusion 最后不做1*1卷积 返回 【深度，RGB】 列表
class CAFuse(nn.Module):
    def __init__(self, c1, reduction=32):
        super().__init__()

        self.get_weights = CAWeights(c1, c1, reduction)


    def forward(self, x):
        h_weights, w_weights = self.get_weights(x)
        out_x1 = x[0] + x[1] * w_weights[1] * h_weights[1]
        out_x2 = x[1] + x[0] * w_weights[0] * h_weights[0]
        out = [out_x1, out_x2]
        return out
        # return x

class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(dim, ouc, k=1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x


    def forward_split_cat(self, x):                 #for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x






class CAFusion(nn.Module):
    def __init__(self, c1, reduction=32):
        super().__init__()

        self.get_weights = CAWeights(c1, c1, reduction)
        self.norm = nn.BatchNorm2d(c1)
        self.part1 = c1 // 4
        self.part2 = c1 - self.part1
        self.cv1 = Conv(2 * c1, c1, 1, 1)
        self.cv2 = nn.Conv2d(self.part1, self.part1, 3, 1, 1, bias=False)
        self.cv3 = Conv(c1, c1//2, 1, 1)
        self.cv4 = Conv(c1//2, c1, 1, 1)

    def forward(self, x):
        # h_weights, w_weights = self.get_weights(x)
        # out_x1 = x[0] + x[1] * w_weights[1] * h_weights[1]
        # out_x2 = x[1] + x[0] * w_weights[0] * h_weights[0]
        # # out = self.cv1(torch.cat((out_x1, out_x2), dim=1))               #直接1*1
        # # out = out_x2 + self.cv1(torch.cat((out_x1, out_x2), dim=1))      #加残差
        # # out = self.norm(out)
        # shortcut = self.cv1(torch.cat((out_x1, out_x2), dim=1))
        # x1, x2 = torch.split(shortcut, [self.part1, self.part2], dim=1)
        # x1 = self.cv2(x1)
        # out_res = torch.cat((x1, x2), 1)
        # out_res = self.cv4(self.cv3(out_res))
        # out = self.norm(shortcut + out_res)
        # return out
        return x[0]



#双流1*1卷积，将融合结果降维后送到FPA进行特征融合
class PointwiseConv(nn.Module):
    def __init__(self, c1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.cv1 = Conv(2 * c1, c1, 1, 1)

    def forward(self, x):
        out = self.cv1(torch.cat((x[0], x[1]), dim=1))
        return out
        # return x[1]

class ConvAWS2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.register_buffer('weight_gamma', torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('weight_beta', torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = weight / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x):
        weight = self._get_weight(self.weight)
        return super()._conv_forward(x, weight, None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.weight_gamma.data.fill_(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            return
        weight = self.weight.data
        weight_mean = weight.data.mean(dim=1, keepdim=True).mean(dim=2,
                                                                 keepdim=True).mean(dim=3, keepdim=True)
        self.weight_beta.data.copy_(weight_mean)
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)


class SAConv2d(ConvAWS2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 s=1,
                 p=None,
                 g=1,
                 d=1,
                 act=True,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=s,
            padding=autopad(kernel_size, p, d),
            dilation=d,
            groups=g,
            bias=bias)
        self.switch = torch.nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=1,
            stride=s,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        out_s = super()._conv_forward(x, weight, None)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        out_l = super()._conv_forward(x, weight, None)
        out = out_s + switch * out_l
        # out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        # avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        # avg_x = self.post_context(avg_x)
        # avg_x = avg_x.expand_as(out)
        # out = out + avg_x
        return self.act(self.bn(out))


class Bottleneck_SAConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = SAConv2d(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_SAConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_SAConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x = self.cv1(x)
        x = x.chunk(2, 1)
        y = list(x)
        # y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3_SA(nn.Module):  # 5.0版本模型backbone和head用的都是这个
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck_SAConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)])  # n个残差组件(Bottleneck)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))



import torchvision

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

# class deformable_LKA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
#         self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
#         self.conv1 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn = self.conv_spatial(attn)
#         attn = self.conv1(attn)
#         return u * attn

class deformable_LKA(nn.Module):
    def __init__(self, c1):
        super().__init__()
        c_ = c1 // 2
        self.conv0 = DeformConv(c_, kernel_size=(5, 5), padding=2, groups=c_)
        self.conv_spatial = DeformConv(c_, kernel_size=(7, 7), stride=1, padding=9, groups=c_, dilation=3)
        self.cv0 = DeformConv(c_, kernel_size=(3, 3), padding=1, groups=c_)
        # self.cv0 = DeformConv(c_, kernel_size=(3, 3), padding=3, groups=1, dilation=3)
        self.cv_spatial = DeformConv(c_, kernel_size=(5, 5), stride=1, padding=6, groups=c_, dilation=3)
        self.conv1 = nn.Conv2d(c_, c_, 1)
        self.conv2 = nn.Conv2d(c_, c_, 1)

    def forward(self, x):
        u = x.clone()
        x = x.chunk(2, 1)
        y = list(x)

        attn0 = self.cv0(y[0])
        attn0 = self.cv_spatial(attn0)
        attn0 = self.conv1(attn0)

        attn1 = self.conv0(y[1])
        attn1 = self.conv_spatial(attn1)
        attn1 = self.conv2(attn1)
        # attn1 = attn1 * y[1]

        attn = torch.cat((attn0, attn1), dim=1)
        # attn = self.conv1(attn)
        return u * attn
        # return attn

# class M_DLK(nn.Module):
#     def __init__(self, c1):
#         super().__init__()
#         self.c_ = c1 // 2
#         self.cv0 = DeformConv(self.c_, kernel_size=(3, 3), stride=1,
#                               padding=autopad(k=3, d=1), groups=self.c_, dilation=1)
#         self.cv1 = DeformConv(self.c_, kernel_size=(5, 5), stride=1,
#                               padding=autopad(k=5, d=3), groups=self.c_, dilation=3)
#         self.cv2 = DeformConv(self.c_, kernel_size=(5, 5), stride=1,
#                               padding=autopad(k=5, d=1), groups=self.c_, dilation=1)
#         self.cv3 = DeformConv(self.c_, kernel_size=(7, 7), stride=1,
#                               padding=autopad(k=7, d=3), groups=self.c_, dilation=3)
#         self.cv4 = nn.Conv2d(c1, c1, 1)
#         self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
#         self.bn = nn.BatchNorm2d(c1)
#         self.act = nn.SiLU()
#
#     def forward(self, x):
#         u = x.clone()
#         x = x.chunk(2, 1)
#         y = list(x)
#         y[0] = self.cv1(self.cv0(y[0]))
#         y[1] = self.cv3(self.cv2(y[1]))
#         out = self.cv4(u * torch.cat(y, 1))
#         # out = self.bn(u + shorcut)
#         # x = x.chunk(4, 1)
#         # y = list(x)
#         # y[1] = self.cv0(y[1])
#         # y[2] = self.cv1(y[2])
#         # y[3] = self.cv3(self.cv2(x[3]))
#         # out = u * self.cv4(torch.cat(y, 1))
#
#         # return self.act(self.bn(out))
#         return out

class P_DLK(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.c_ = c1 // 4
        self.spare = c1 - self.c_
        self.conv0 = DeformConv(self.c_, kernel_size=(5, 5), padding=2, groups=1)
        self.conv_spatial = DeformConv(self.c_, kernel_size=(7, 7), stride=1, padding=9, groups=1, dilation=3)
        # self.conv0 = DeformConv(self.c_, kernel_size=(3, 3), padding=1, groups=1)
        # self.conv_spatial = DeformConv(self.c_, kernel_size=(5, 5), stride=1, padding=6, groups=1, dilation=3)
        self.cv1 = Conv(c1, c2, k, s)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.c_, self.spare], dim=1)
        x1 = self.conv0(x1)
        x1 = self.conv_spatial(x1)

        x = torch.cat((x1, x2), 1)
        x = self.cv1(x)

        return x


class Bottleneck_DLKA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = deformable_LKA(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_DLKA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_DLKA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x = self.cv1(x)
        x = x.chunk(2, 1)
        y = list(x)
        # y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3_DLKA(nn.Module):  # 5.0版本模型backbone和head用的都是这个
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck_DLKA(c_, c_, shortcut, g, e=1.0) for _ in range(n)])  # n个残差组件(Bottleneck)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)) # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


