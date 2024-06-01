from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


def get_upsample_filter(size: int) -> Tensor:
    """Make a 2D bilinear kernel suitable for upsampling.

    Args:
        size (int): The size of the filter.

    Returns:
        Tensor: A 2D bilinear kernel.
    """
    factor = (size + 1) // 2
    center = (factor - 1) if size % 2 == 1 else (factor - 0.5)
    og = np.ogrid[:size, :size]
    upsample_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(upsample_filter).float()


def initialize_weights(
    module: Union[nn.Module, nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d],
) -> None:
    """Initialize weights for different types of layers.

    Args:
        module (Union[nn.Module, nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d]): The module to initialize.
    """
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

    elif isinstance(module, nn.ConvTranspose2d):
        c1, c2, h, w = module.weight.data.size()
        weight = get_upsample_filter(h)
        module.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
        if module.bias is not None:
            module.bias.data.zero_()
