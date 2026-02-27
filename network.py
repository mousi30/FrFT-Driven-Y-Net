# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch_frft.dfrft_module import dfrft, idfrft
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from block import CustomIntegratedBlock, skip_connection, CRU
from dysample import UpsampleAndConv







class SwinFusion(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4],
                 Ex_num_heads=[6], Fusion_num_heads=[6, 6], Re_num_heads=[6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinFusion, self).__init__()
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        print('in_chans: ', in_chans)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)



        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim_temp, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################


        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio


        # merge non-overlapping patches into image
        self.softmax = nn.Softmax(dim=0)
        # absolute position embedding

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers_Fusion = nn.ModuleList()
        self.block = CustomIntegratedBlock(in_channels=1, channel_list=[4, 8, 16, 24, 32], kernel_size=3, stride=1)
        self.cru_fusion = CRU(64)
        self.skips = nn.ModuleDict({
            '4': skip_connection(4),
            '8': skip_connection(8),
            '16': skip_connection(16),
            '32': skip_connection(32),
            '48': skip_connection(48),
            '64': skip_connection(64),
            '128': skip_connection(128),
            '256': skip_connection(256),
            '512': skip_connection(512),
            '1024': skip_connection(1024),
            # 预先定义更多可能的通道数配置
        })
        self.Upsampling = UpsampleAndConv([32, 24, 16, 8, 4, 1])
        self.conv_fusion = nn.Conv2d(64, 32, 3, 1, 1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # def check_image_size(self, x):
    #     _, _, h, w = x.size()
    #     mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
    #     mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
    #     x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    #     return x

    def skip_process(self, skip_a, skip_b):
        skip = [torch.cat((a, b), dim=1) for a, b in zip(skip_a, skip_b)]
        processed_feature_maps = []
        for skip in skip:
            # 假设 feature_map 的通道数是我们可以处理的
            num_channels = skip.size(1)  # 获取特征图的通道数
            if str(num_channels) in self.skips:
                # 通过相应的 skip_connection 处理特征图
                processed_map = self.skips[str(num_channels)](skip)
                processed_feature_maps.append(processed_map)
            else:
                # 如果没有对应的处理函数，可能需要某种形式的处理或者警告
                print(f"No skip connection available for {num_channels} channels.")
        return processed_feature_maps

    def forward_features_Ex_A(self, x):
        x, fre_map = self.block(x)
        return x, fre_map

    def forward_features_Ex_B(self, x):
        x, fre_map = self.block(x)
        return x, fre_map

    def forward_features_Fusion(self, x, y):
        x = torch.cat((x, y), 1)
        x = self.cru_fusion(x)
        x = self.cru_fusion(x)
        ## Downsample the feature in the channel dimension
        x = self.lrelu(self.conv_fusion(x))
        return x

    def forward_features_Re(self, x, skip):
        x = self.Upsampling(x, skip)
        return x

    def forward(self, A, B):
        x = A
        y = B
        self.mean_A = self.mean.type_as(x)
        self.mean_B = self.mean.type_as(y)
        self.mean = (self.mean_A + self.mean_B) / 2

        x = (x - self.mean_A) * self.img_range
        y = (y - self.mean_B) * self.img_range
        # Feedforward
        x, skip_a = self.forward_features_Ex_A(x)
        y, skip_b = self.forward_features_Ex_B(y)
        skip = self.skip_process(skip_a, skip_b)
        x = self.forward_features_Fusion(x, y)
        x = self.forward_features_Re(x, skip)

        x = x / self.img_range + self.mean
        return x



if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 520
    width = 520
    model = SwinFusion(upscale=2, img_size=(height, width), in_chans=1,
                       window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                       embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='')
    # print(model)
    # print(height, width, model.flops() / 1e9)

    x = torch.randn(9, 1, height, width)
    y = torch.randn(9, 1, height, width)
    x = model(x, y)
    print(x.shape)
