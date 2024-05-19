#!/usr/bin/env python3
"""
Adapted from
https://github.com/Tramac/awesome-semantic-segmentation-pytorch
"""

import logging
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from models import encoder as enc

logger = logging.getLogger("pytorch_lightning")

__all__ = ["SegBaseModel", "initialize_weights", "print_network"]


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class SegBaseModel(nn.Module):
    def __init__(
        self, nclass, backbone="ResNet50", pretrained_base=True, dilated=None, **kwargs
    ):
        super(SegBaseModel, self).__init__()

        if dilated is None:
            dilated = True

        self.nclass = nclass
        model_name = backbone.lower()
        model = enc.__dict__[model_name.lower()]
        self.encoder = model(pretrained=pretrained_base, dilated=dilated, **kwargs)

    def base_forward(self, x, multiscale=False):
        """Forwarding pre-trained network."""
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
        else:
            return [c1, c2, c3, c4]


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling."""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.ConvTranspose2d):
        c1, c2, h, w = module.weight.data.size()
        weight = get_upsample_filter(h)
        module.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
        if module.bias is not None:
            module.bias.data.zero_()


def check_mismatch(model, pretrained_dict) -> None:
    model_dict = model.state_dict()
    temp_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            if model_dict[k].shape != pretrained_dict[k].shape:
                logger.info(
                    f"Skip loading parameter: {k}, "
                    f"required shape: {model_dict[k].shape}, "
                    f"loaded shape: {pretrained_dict[k].shape}"
                )
                continue
            else:
                temp_dict[k] = v
    return temp_dict


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(f"Total number of parameters: {int(num_params) / 1000**2} M")
