#!/usr/bin/env python

#####################################################################################################################################################################
# FTNet                                                                                                                                                             #
# Copyright 2020 Tufts University.                                                                                                                                  #                                                                                                                #
# Please see LICENSE file for full terms.                                                                                                                           #                                                                                                                                              #                                                                                                                                                 #
#####################################################################################################################################################################

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("pytorch_lightning")
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        previous_dilation=1,
        downsample=None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            3,
            1,
            previous_dilation,
            dilation=previous_dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class FeatureTransverseDecoder(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        dilation,
        edge_extract=None,
    ):
        super().__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels, dilation
        )

        self.num_inchannels = num_inchannels.copy()
        self.num_branches = num_branches
        self.dilation = dilation
        self.edge_extract = edge_extract

        edge_list = []
        for i in edge_extract:
            edge_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.num_inchannels[i],
                        out_channels=1,
                        kernel_size=3,
                        stride=1,
                        padding=3 // 2,
                    ),
                    nn.BatchNorm2d(1, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
            )

        self.edges = nn.ModuleList(edge_list)

        self.final_edge = nn.Sequential(
            nn.Conv2d(
                in_channels=len(edge_extract),
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
            )
        )

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels, dilation
    ):
        if num_branches != len(num_blocks):
            error_msg = f"NUM_BRANCHES({num_branches}) NOT EQUAL TO NUM_BLOCKS({len(num_blocks)})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = (
                f"NUM_BRANCHES({num_branches})  NOT EQUAL TO NUM_CHANNELS({len(num_channels)})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = (
                f"NUM_BRANCHES({num_branches})  NOT EQUAL TO NUM_INCHANNELS({len(num_inchannels)})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(dilation):
            error_msg = f"NUM_BRANCHES({num_branches})  NOT EQUAL TO Dilation({len(dilation)})"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=0.01),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.num_inchannels[branch_index],
                planes=num_channels[branch_index],
                stride=stride,
                downsample=downsample,
                dilation=self.dilation[branch_index],
                previous_dilation=self.dilation[branch_index],
            )
        )

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    inplanes=self.num_inchannels[branch_index],
                    planes=num_channels[branch_index],
                    dilation=self.dilation[branch_index],
                    previous_dilation=self.dilation[branch_index],
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=0.01),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    previous_dilated_branches = (
                        np.array(self.dilation[-(num_branches - j - 1) :]) > 1
                    )
                    current_dilated_branches = bool(self.dilation[i] > 1)
                    for branch_index in range(i - j):
                        if branch_index == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            if current_dilated_branches:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2d(
                                            in_channels=num_inchannels[j],
                                            out_channels=num_outchannels_conv3x3,
                                            kernel_size=3,
                                            stride=1,
                                            padding=self.dilation[i],
                                            dilation=self.dilation[i],
                                            bias=False,
                                        ),
                                        nn.BatchNorm2d(num_outchannels_conv3x3, momentum=0.01),
                                    )
                                )
                            else:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2d(
                                            in_channels=num_inchannels[j],
                                            out_channels=num_outchannels_conv3x3,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False,
                                        ),
                                        nn.BatchNorm2d(num_outchannels_conv3x3, momentum=0.01),
                                    )
                                )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            if previous_dilated_branches[branch_index]:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2d(
                                            in_channels=num_inchannels[j],
                                            out_channels=num_outchannels_conv3x3,
                                            kernel_size=3,
                                            stride=1,
                                            padding=self.dilation[-(num_branches - j - 1) :][
                                                branch_index
                                            ],
                                            dilation=self.dilation[-(num_branches - j - 1) :][
                                                branch_index
                                            ],
                                            bias=False,
                                        ),
                                        nn.BatchNorm2d(num_outchannels_conv3x3, momentum=0.01),
                                        nn.ReLU(inplace=True),
                                    )
                                )

                            else:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2d(
                                            in_channels=num_inchannels[j],
                                            out_channels=num_outchannels_conv3x3,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False,
                                        ),
                                        nn.BatchNorm2d(num_outchannels_conv3x3, momentum=0.01),
                                        nn.ReLU(inplace=True),
                                    )
                                )

                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x, h, w):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        edge_feats = []
        j = 0
        for i in range(self.num_branches):
            if i in self.edge_extract:
                temp = self.edges[j](x[i])
                edge_feats.append(
                    F.interpolate(temp, size=(h, w), mode="bilinear", align_corners=True)
                )
                j += 1

            x[i] = self.branches[i](x[i])

        edge = self.final_edge(torch.cat(edge_feats, 1))

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])

            x_fuse.append(
                F.interpolate(self.relu(y), size=(h, w), mode="bilinear", align_corners=True)
            )
        return x_fuse, edge
