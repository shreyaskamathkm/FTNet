"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

import logging
import os
from pathlib import Path
from typing import List

from .file_path_handler import FilePathHandler
from .segbase import SegmentationDataset
from .transforms import NormalizationTransform, ScutsegTransform

logger = logging.getLogger(__name__)


class SCUTSEGDataset(SegmentationDataset):
    NUM_CLASS = 10
    IGNORE_INDEX = 255
    NAME = "scutseg"
    mean = [0.41213047, 0.42389206, 0.416051]
    std = [0.13611181, 0.13612076, 0.13611817]

    def __init__(
        self,
        root: Path,
        split: str = "train",
        base_size: List[List[int]] = [[520, 520]],
        crop_size: List[List[int]] = [[480, 480]],
        mode: str = None,
        sobel_edges: bool = False,
    ):
        super().__init__(root, split, mode, base_size, crop_size, sobel_edges)

        if self.mode == "test":
            self.mode = "val"  # ScutSEG dataset does not have validation

        self.images, self.mask_paths, self.edge_paths = FilePathHandler._get_pairs(
            self.root, self.split
        )

        if len(self.images) != len(self.mask_paths):
            raise ValueError("Mismatch between images and masks")
        if len(self.images) != len(self.edge_paths):
            raise ValueError("Mismatch between images and edges")
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in subfolder of: {root}")

        valid_classes = [0, 7, 24, 25, 26, 27, 13, 21, 28, 17]
        class_map = dict(zip(valid_classes, range(10)))
        self.update_normalization(NormalizationTransform(self.mean, self.std))
        self.update_image_transform(
            ScutsegTransform(valid_classes=valid_classes, class_map=class_map)
        )

    @property
    def class_names(
        self,
    ):
        return [
            "background",
            "road",
            "person",
            "rider",
            "car",
            "truck",
            "fence",
            "tree",
            "bus",
            "pole",
        ]


if __name__ == "__main__":
    from os import path, sys

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(os.path.join(path.dirname(path.dirname(path.abspath(__file__))), ".."))
    from core.data.samplers import make_data_sampler, make_multiscale_batch_data_sampler
    from torch.utils.data import DataLoader

    from datasets import *

    data_kwargs = {"base_size": [520], "crop_size": [480]}

    train_dataset = SCUTSEGDataset(
        root="/mnt/ACF29FC2F29F8F68/Work/Deep_Learning/Thermal_Segmentation/Dataset/",
        split="train",
        mode="train",
        **data_kwargs,
    )

    train_sampler = make_data_sampler(dataset=train_dataset, shuffle=False, distributed=False)

    train_batch_sampler = make_multiscale_batch_data_sampler(
        sampler=train_sampler, batch_size=1, multiscale_step=1, scales=1
    )

    loader_random_sample = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        pin_memory=True,
    )

    x = train_dataset.__getitem__(200)
