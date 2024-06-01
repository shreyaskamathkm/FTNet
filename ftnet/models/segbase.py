#!/usr/bin/env python3
"""
Adapted from
https://github.com/Tramac/awesome-semantic-segmentation-pytorch
"""

import logging
import sys
from typing import List, Optional

import torch.nn as nn
from torch import Tensor

from models import encoder as enc

logger = logging.getLogger(__name__)

__all__ = ["SegBaseModel"]


def str_to_class(classname: str) -> type:
    """Convert a string to a class object.

    Args:
        classname (str): The name of the class.

    Returns:
        type: The class object.
    """
    return getattr(sys.modules[__name__], classname)


class SegBaseModel(nn.Module):
    def __init__(
        self,
        nclass: int,
        backbone: str = "ResNet50",
        pretrained_base: bool = True,
        dilated: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """Segmentation Base Model Initialization.

        Args:
            nclass (int): Number of output classes.
            backbone (str, optional): Backbone model name. Defaults to "ResNet50".
            pretrained_base (bool, optional): Whether to use pretrained weights for the base. Defaults to True.
            dilated (Optional[bool], optional): Whether to use dilated convolutions. Defaults to True.
        """
        super().__init__()

        self.nclass = nclass
        model_name = backbone.lower()
        model = enc.__dict__[model_name]
        self.encoder = model(pretrained=pretrained_base, dilated=dilated, **kwargs)

    def base_forward(self, x: Tensor, multiscale: bool = False) -> List[Tensor]:
        """Forwarding through the pre-trained network.

        Args:
            x (Tensor): Input tensor.
            multiscale (bool, optional): Whether to return multiscale features. Defaults to False.

        Returns:
            List[Tensor]: List of feature maps.
        """
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        c0 = self.encoder.relu(x)
        x = self.encoder.maxpool(c0)
        c1 = self.encoder.layer1(x)
        c2 = self.encoder.layer2(c1)
        c3 = self.encoder.layer3(c2)
        c4 = self.encoder.layer4(c3)

        if multiscale:
            return [c0, c1, c2, c3, c4]

        return [c1, c2, c3, c4]
