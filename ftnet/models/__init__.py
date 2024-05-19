#!/usr/bin/env python3

from typing import Any

import models.segmentation_decoder as seg_dec

__all__ = ["get_segmentation_model"]


def get_segmentation_model(model_name: str, **kwargs: Any):
    model_func = getattr(seg_dec, f"get_{model_name.lower()}")
    return model_func(**kwargs)


if __name__ == "__main__":
    model = get_segmentation_model("ftnet")
