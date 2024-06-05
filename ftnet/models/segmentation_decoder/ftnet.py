#!/usr/bin/env python

#####################################################################################################################################################################
# FTNet                                                                                                                                                             #
# Copyright 2020 Tufts University.                                                                                                                                  #                                                                                                                #
# Please see LICENSE file for full terms.                                                                                                                           #                                                                                                                                              #                                                                                                                                                 #
#####################################################################################################################################################################
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..base_model import SegBaseModel
from ..model_helper import initialize_weights
from .dcc_net_decoder import BasicBlock, FeatureTransverseDecoder

logger = logging.getLogger(__name__)

__all__ = ["get_ftnet", "FNet"]


class FNet(SegBaseModel):
    def __init__(
        self,
        nclass: int,
        backbone: str = "resnet50",
        pretrained_base: bool = True,
        no_of_filters: int = 32,
        edge_extracts: List[int] = None,
        num_blocks: int = None,
        dilated: bool = False,
        **kwargs,
    ) -> None:
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
        edge_extract = [x - 1 for x in edge_extracts] if edge_extracts else None

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
) -> FNet:
    from ...data import datasets

    return FNet(
        nclass=datasets[dataset.lower()].NUM_CLASS,
        backbone=backbone,
        pretrained_base=pretrained_base,
        **kwargs,
    )
