#!/usr/bin/env python3

from typing import Any

from . import segmentation_decoder as seg_dec


def get_segmentation_model(model_name: str, **kwargs: Any):
    model_func = getattr(seg_dec, f"get_{model_name.lower()}")
    return model_func(**kwargs)
