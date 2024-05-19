#!/usr/bin/env python

#####################################################################################################################################################################
# FTNet                                                                                                                                                             #
# Copyright 2020 Tufts University.                                                                                                                                  #                                                                                                                #
# Please see LICENSE file for full terms.                                                                                                                           #                                                                                                                                              #                                                                                                                                                 #
#####################################################################################################################################################################

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..segbase import SegBaseModel, initialize_weights
from .dcc_net_decoder import BasicBlock, FeatureTransverseDecoder

logger = logging.getLogger("pytorch_lightning")

__all__ = ["FNet", "get_ftnet"]


class FNet(SegBaseModel):
    def __init__(
        self,
        nclass: int,
        backbone: str = "resnet50",
        pretrained_base: bool = True,
        no_of_filters: int = 32,
        edge_extracts: list[int] = None,
        num_blocks: int = None,
        dilated: bool = False,
        **kwargs,
    ):
        super().__init__(
            nclass=nclass,
            backbone=backbone,
            pretrained_base=pretrained_base,
            dilated=dilated,
            **kwargs,
        )

        num_branches = 4
        numblocks = [num_blocks] * num_branches
        num_channels = [no_of_filters] * num_branches
        edge_extract = [x - 1 for x in edge_extracts]

        dilations = [1, 1, 2, 4] if dilated else [1, 1, 1, 1]

        self.decoder = FeatureTransverseDecoder(
            num_branches=num_branches,
            blocks=BasicBlock,
            num_blocks=numblocks,
            num_inchannels=self.encoder.feature_list[1:],
            num_channels=num_channels,
            dilation=dilations,
            edge_extract=edge_extract,
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=np.sum(num_channels) + 1,
                out_channels=np.sum(num_channels) // 2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(np.sum(num_channels) // 2, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=np.sum(num_channels) // 2,
                out_channels=nclass,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.__setattr__("exclusive", ["decoder", "last_layer"])

        if pretrained_base:
            for name, mod in self.named_children():
                if name != "encoder":
                    for m in mod.modules():
                        initialize_weights(m)
        else:
            for m in self.modules():
                initialize_weights(m)

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape
        out, edge = self.decoder(self.base_forward(x, multiscale=False), h, w)
        out.append(edge)

        class_maps = self.last_layer(torch.cat(out, 1))
        return class_maps, edge


def get_ftnet(
    dataset: str = "soda",
    backbone: str = "resnet50",
    root: Path = None,
    pretrained_base: bool = False,
    **kwargs,
):
    from data import datasets

    model = FNet(
        nclass=datasets[dataset.lower()].NUM_CLASS,
        backbone=backbone,
        pretrained_base=pretrained_base,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    from helper import print_network
    from lightning.pytorch.utilities import measure_flops

    img = torch.randint(3, 5, (2, 3, 512, 512)).float()

    with torch.device("meta"):
        model = get_ftnet(
            pretrained_base=False,
            edge_extracts=[3],
            num_blocks=4,
            no_of_filters=32,
            backbone="resnext101_32x8d",
        )
        print_network(model)
        model_fwd = lambda: model(img)
        macs, params = measure_flops(model, model_fwd)
        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))
