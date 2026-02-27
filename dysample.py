# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from block import basic_block


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='xy')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='xy')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


# class UpsampleAndConv(nn.Module):
#     def __init__(self, channels_list):
#         super(UpsampleAndConv, self).__init__()
#         self.layers = nn.ModuleList()
#         in_channels = channels_list[0]
#
#         for out_channels in channels_list[1:]:
#             self.layers.append(DySample(in_channels))
#             self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1))
#             self.layers.append(nn.ReLU(inplace=True))
#             self.layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
#             self.layers.append(nn.ReLU(inplace=True))
#             in_channels = out_channels  # 更新通道数
#
#     def forward(self, x, skip):
#         for layer in self.layers:
#             x = layer(x)
#         return x



class UpsampleAndConv(nn.Module):
    def __init__(self, channels_list):
        super(UpsampleAndConv, self).__init__()
        self.layers = nn.ModuleList()
        self.channels_list = channels_list
        # self.Dysample1 = DySample(channels_list[0])
        self.Dysample_conv1 = nn.Conv2d(channels_list[0], channels_list[1], kernel_size=1, stride=1)
        # self.Dysample2 = DySample(channels_list[1])
        self.Dysample_conv2 = nn.Conv2d(channels_list[1], channels_list[2], kernel_size=1, stride=1)
        # self.Dysample3 = DySample(channels_list[2])
        self.Dysample_conv3 = nn.Conv2d(channels_list[2], channels_list[3], kernel_size=1, stride=1)
        # self.Dysample4 = DySample(channels_list[3])
        self.Dysample_conv4 = nn.Conv2d(channels_list[3], channels_list[4], kernel_size=1, stride=1)

        # 假设 skip 列表的特征图通道数和 channels_list 对齐
        self.upsample_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.conv1 = nn.Conv2d(channels_list[0] * 2, channels_list[0], kernel_size=3, padding=1, stride=1)
        self.conv1_1 = nn.Conv2d(channels_list[0], channels_list[0], kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(channels_list[1] * 2, channels_list[1], kernel_size=3, padding=1, stride=1)
        self.conv2_2 = nn.Conv2d(channels_list[1], channels_list[1], kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(channels_list[2] * 2, channels_list[2], kernel_size=3, padding=1, stride=1)
        self.conv3_3 = nn.Conv2d(channels_list[2], channels_list[2], kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(channels_list[3] * 2, channels_list[3], kernel_size=3, padding=1, stride=1)
        self.conv4_4 = nn.Conv2d(channels_list[3], channels_list[3], kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(channels_list[4] * 2, channels_list[4], kernel_size=3, padding=1, stride=1)
        self.conv5_5 = nn.Conv2d(channels_list[4], channels_list[4], kernel_size=3, padding=1, stride=1)
        self.conv7 = nn.Conv2d(channels_list[4]*2, channels_list[4]*2, kernel_size=7, padding=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        # self.conv_last = nn.Conv2d(channels_list[4], channels_list[4], kernel_size=3,padding=1,stride=1)

    def forward(self, x, skips):
        skips = list(reversed(skips))
        # 通过循环处理每一个上采样和卷积层
        x = torch.cat([x, skips[0]], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1_1(x)
        x = self.relu(x)
        # x = self.Dysample1(x)
        x = self.Dysample_conv1(x)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        # x = self.Dysample2(x)
        x = self.Dysample_conv2(x)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        # x = self.Dysample3(x)
        x = self.Dysample_conv3(x)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv4_4(x)
        x = self.relu(x)
        # x = self.Dysample4(x)
        x = self.Dysample_conv4(x)
        x = torch.cat([x, skips[4]], dim=1)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv5(x)
        return x


if __name__ == '__main__':
    channels_list = [32, 24, 16, 8, 4, 1]  # 包括初始通道数和目标通道数列表
    model = UpsampleAndConv(channels_list)

    # 假设输入的张量 x
    x = torch.rand(64, 32, 4, 4)

    # 应用模型并打印输出形状
    output = model(x)
    print(output.shape)
