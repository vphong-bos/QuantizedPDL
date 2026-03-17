# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Modified from the Panoptic-DeepLab implementation in Detectron2 library
# https://github.com/facebookresearch/detectron2/tree/main/projects/Panoptic-DeepLab
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    """
    PyTorch implementation of BottleneckBlock for ResNet.

    Based on the model structure, BottleneckBlock contains:
    - conv1: 1x1 conv (channel reduction)
    - conv2: 3x3 conv (spatial convolution, potentially with stride/dilation)
    - conv3: 1x1 conv (channel expansion)
    - shortcut: optional 1x1 conv for residual connection when input/output dimensions differ
    Each with SyncBatchNorm and ReLU activation, except the final output uses residual addition + ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        has_shortcut: bool = False,
        shortcut_stride: int = 1,
    ):
        super().__init__()
        self.has_shortcut = has_shortcut

        # Main path convolutions with norm as submodules to match state dict structure
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(bottleneck_channels, eps=1e-05, momentum=0.1)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.SyncBatchNorm(bottleneck_channels, eps=1e-05, momentum=0.1)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.SyncBatchNorm(out_channels, eps=1e-05, momentum=0.1)

        # Shortcut connection with norm as submodule to match state dict structure
        if has_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=shortcut_stride, bias=False)
            self.shortcut_bn = nn.SyncBatchNorm(out_channels, eps=1e-05, momentum=0.1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.has_shortcut:
            identity = self.shortcut(identity)
            identity = self.shortcut_bn(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)

        return out