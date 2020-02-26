#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:14:16 2019

@author: xingyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4,
                      ch_out // 4,
                      kernel_size=(3, 1),
                      padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4,
                      ch_out // 4,
                      kernel_size=(1, 3),
                      padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).permute(0, 2, 1)
        pooled = F.max_pool1d(x, self.kernel_size, self.stride, self.padding,
                              self.dilation, self.ceil_mode,
                              self.return_indices)
        _, _, c_new = pooled.shape
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c_new, h, w)


class MaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride) -> None:
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        # self.channel_pooling = ChannelPool(kernel_size=kernel_size[0],
        #                                    stride=stride[0])
        # self.spatial_pooling = nn.MaxPool2d(kernel_size=kernel_size[1:],
        #                                     stride=stride[1:])

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        x = x.view(n, 1, c, h, w)
        x = torch.max_pool3d(x,
                             kernel_size=self.kernel_size,
                             stride=self.stride)
        n, _, c, h, w = x.shape
        x = x.view(n, c, h, w)
        # x = self.channel_pooling(x)
        # x = self.spatial_pooling(x)
        return x


class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                      stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64,
                      out_channels=256,
                      kernel_size=(1, 4),
                      stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256,
                      out_channels=class_num,
                      kernel_size=(13, 1),
                      stride=1),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=256 + class_num + 128 + 64,
                      out_channels=self.class_num,
                      kernel_size=(1, 1),
                      stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            # f_pow = torch.pow(f, 2)
            # f_pow = torch.mul(f, f)
            if self.training:
                f_pow = torch.pow(f, 2)
                f_mean = torch.mean(f_pow)
            else:
                f_pow = torch.abs(f) * torch.abs(f)
                f_mean = torch.mean(f_pow).item()
            # f_mean = F.mse_loss(f, torch.zeros_like(f))
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '皖', '闽', '赣',
    '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z', 'I', 'O', '-'
]

if __name__ == "__main__":

    # from torchsummary import summary
    #
    # lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    # print(lprnet)
    #
    # summary(lprnet, (3, 24, 94), device="cpu")

    a = torch.randn(1, 16, 4, 4)
    module = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2))
    module_2 = MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2))
    print(module(a).shape)
    print(module_2(a).shape)
