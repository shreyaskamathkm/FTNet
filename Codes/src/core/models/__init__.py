#!/usr/bin/env python3
try:
    import segmentation_decoder as seg_dec
except:
    import core.models.segmentation_decoder as seg_dec
from typing import Any

__all__ = ['get_segmentation_model']


def get_segmentation_model(model_name: str, **kwargs: Any):

    model = seg_dec.__dict__['get_' + model_name.lower()]
    return model(**kwargs)


if __name__ == '__main__':
    model = get_segmentation_model('ftnet')
