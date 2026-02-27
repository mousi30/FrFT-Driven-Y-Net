# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch_frft.dfrft_module import dfrft, idfrft
import torch.nn.functional as F
from defconv import DefC


class DoubleDefCBlock(nn.Module):
    def __init__(self, inc, mid_channels, outc, kernel_size=3, padding=1, stride=1, bias=None):
        super(DoubleDefCBlock, self).__init__()
        self.defc1 = DefC(inc, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.defc2 = DefC(mid_channels, outc, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.defc3 = DefC(outc, outc, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        # self.defc4 = DefC(mid_channels, outc, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.conv = nn.Conv2d(inc, outc, 1, stride)
        # self.att = selfAttention(outc, outc)

    def forward(self, x):
        origin = self.conv(x)
        x = self.defc1(x)
        x = self.relu(x)
        x = self.defc2(x)
        x = self.relu(x)
        x = self.defc3(x)
        # x = self.relu(x)
        # x = self.defc4(x)
        # x = self.att(x, x)
        return x + origin


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False  # 如果同时使用偏置和归一化，需要关闭偏置
        padding = kernel_size // 2
        layers = list()
        # 添加标准的卷积层
        layers.append(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            layers.append(nn.LeakyReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ConvExtractionBlock(nn.Module):
    def __init__(self, inout_channels, mid_channels, kernel_size, stride=1, order=0.5, dim=-1, trainable=True):
        super().__init__()
        self.order = nn.Parameter(torch.tensor([order], dtype=torch.float32), requires_grad=trainable)
        self.dim = dim
        self.dfrft = lambda x: dfrft(x, self.order, dim=self.dim)
        self.idfrft = lambda x: idfrft(x, self.order, dim=self.dim)
        self.main_fft = nn.Sequential(
            BasicConv(inout_channels*2, mid_channels*2, kernel_size, stride, relu=True),
            BasicConv(mid_channels*2, inout_channels*2, kernel_size, stride, relu=False)
        )

    def forward(self, x):
        origin = x
        x = self.dfrft(x)
        # x = torch.fft.rfft2(x)
        real = x.real
        imag = x.imag
        x = torch.cat([real, imag], dim=1)  # 在通道维度上连接
        x = self.main_fft(x)
        real, imag = torch.chunk(x, 2, dim=1)  # 在通道维拆分实部和虚部
        x = torch.complex(real, imag)
        x = self.idfrft(x)
        x = torch.abs(x)
        return x + origin


class invertedBlock(nn.Module):
    def __init__(self, in_channel,ratio=2):
        super(invertedBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.GELU()
        ## 7*7卷积，并行3*3卷积
        self.conv1 = nn.Conv2d(internal_channel, internal_channel, 7, 1, 3, groups=in_channel,bias=False)

        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=in_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.pw1 = nn.Conv2d(in_channels=in_channel, out_channels=internal_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channel, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)

    def hifi(self,x):
        x1=self.pw1(x)
        x1=self.relu(x1)
        x1=self.conv1(x1)
        x1=self.relu(x1)
        x1=self.pw2(x1)
        x1=self.relu(x1)
        x3 = x1+x
        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.layer_norm(x3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = self.convFFN(x3)
        return x4

    def forward(self, x):
        return self.hifi(x)+x


class ConvFFN(nn.Module):

    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()
        internal_channels = in_channels * expend_ratio
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        x1 = self.pw1(x)
        x2 = self.nonlinear(x1)
        x3 = self.pw2(x2)
        x4 = self.nonlinear(x3)
        return x4 + x


class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False),nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.alpha*self.conv1(x)+self.beta*self.conv2(x)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        if channel > 3:
            reduction = 8
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Downupblock(nn.Module):
    def __init__(self, n_feats):
        super(Downupblock, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)
        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))
        self.alise = nn.Conv2d(n_feats, n_feats, 1, 1, 0, bias=False)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1, bias=False)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))
        self.raw_alpha.data.fill_(0)
        self.ega=selfAttention(n_feats, n_feats)

    def forward(self, x, raw):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        high=high+self.ega(high,high)*self.raw_alpha
        x2=self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x


class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)  # nn.Sequential(one_module(n_feats),
        #                     one_module(n_feats),
        #                     one_module(n_feats))
        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))

        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))
        # fill 0
        self.raw_alpha.data.fill_(0)
        self.ega=selfAttention(n_feats, n_feats)
        self.sf = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=2, dilation=2)

    def forward(self, x, raw):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        high=high+self.ega(high,high)*self.raw_alpha
        x2=self.decoder_low(x2)
        x3 = x2
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x


class selfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(selfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, feature, feature_map):
        query = self.query_conv(feature)
        key = self.key_conv(feature)
        value = self.value_conv(feature)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        output_feature_map = (feature_map + attended_values)
        return output_feature_map


class basic_block(nn.Module):
    ## 双并行分支，通道分支和空间分支
    def __init__(self, in_channel, depth, ratio=1):
        super(basic_block, self).__init__()
        self.rep1 = nn.Sequential(
            *[invertedBlock(in_channel=in_channel, ratio=ratio) for i in range(depth)])
        self.relu = nn.GELU()
        # 一部分做3个3*3卷积，一部分做1个
        self.updown = Updownblock(in_channel)
        self.downup = Downupblock(in_channel)

    def forward(self, x, raw=None):
        x1 = self.rep1(x)
        x1 = self.updown(x1, raw)
        x1 = self.downup(x1, raw)
        return x1 + x



class downBlock(nn.Module):
    def __init__(self, in_channels, channel_list, kernel_size, stride, ratio=2):
        super(downBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.skip_connections = []  # 用于存储跳跃连接的特征图

        previous_channels = in_channels
        for i, current_out_channels in enumerate(channel_list):
            self.layers.append(DoubleDefCBlock(
                inc=previous_channels,
                mid_channels=previous_channels,
                outc=current_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride
            ))
            previous_channels = current_out_channels
            # 将每个深度的输出添加到跳跃连接列表
            self.skip_connections.append(len(self.layers) - 1)

    def forward(self, x):
        skip_features = [x]
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.skip_connections:
                skip_features.append(x)
        return x, skip_features



class DOConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DOConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        origin = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + origin


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, depth):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.gp1 = basic_block(in_channel=mid_channels, depth=1, ratio=2)
        self.gp2 = basic_block(in_channel=mid_channels, depth=1, ratio=2)
        self.gp3 = basic_block(in_channel=mid_channels, depth=1, ratio=2)
        self.conv_first1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_first1_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.layers1 = nn.ModuleList([DOConvBlock(mid_channels, mid_channels) for _ in range(depth)])
        self.layers2 = nn.ModuleList([DOConvBlock(mid_channels, mid_channels) for _ in range(depth)])
        self.layers3 = nn.ModuleList([DOConvBlock(mid_channels, mid_channels) for _ in range(depth)])
        self.conv_last1 = nn.Conv2d(mid_channels*2, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_last2 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pt_conv1 = ConvExtractionBlock(inout_channels=in_channels, mid_channels=in_channels,
                                           kernel_size=1, stride=1, order=1, trainable=True)
        self.pt_conv2 = ConvExtractionBlock(inout_channels=in_channels, mid_channels=in_channels,
                                           kernel_size=1, stride=1, order=1, trainable=True)

    def forward(self, x, y):

        x = self.pt_conv1(x)
        y = self.pt_conv2(y)
        x = self.conv_first1(x)
        x = self.relu(x)
        y = self.conv_first1_1(y)
        y = self.relu(y)
        for layer in self.layers1:
            x = layer(x)
        x = self.gp1(x)
        for layer in self.layers2:
            y = layer(y)
        y = self.gp2(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv_last1(x)
        x = self.gp3(x)
        for layer in self.layers3:
            x = layer(x)
        x = self.conv_last2(x)
        return x


