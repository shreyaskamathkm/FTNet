#!/usr/bin/env python3

from typing import Any, Union

from .segmentation_loss import EdgeNetLoss, MixSoftmaxCrossEntropyLoss

__all__ = ["get_segmentation_loss"]


def get_segmentation_loss(
    model: str, **kwargs: Any
) -> Union[MixSoftmaxCrossEntropyLoss, EdgeNetLoss]:
    """Get the appropriate segmentation loss function based on the model name.

    Args:
        model (str): Name of the model.
        **kwargs: Additional arguments for the loss function.

    Returns:
        Union[MixSoftmaxCrossEntropyLoss, EdgeNetLoss]: The loss function.
    """
    model = model.lower()
    if "ftnet" in model:
        return EdgeNetLoss(**kwargs)
    return MixSoftmaxCrossEntropyLoss(**kwargs)
