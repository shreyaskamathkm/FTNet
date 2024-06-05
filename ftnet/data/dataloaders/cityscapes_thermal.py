"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

import logging
import os
from pathlib import Path
from typing import List

import numpy as np

from .file_path_handler import FilePathHandler
from .segbase import SegmentationDataset
from .transforms import CityscapeTransform, NormalizationTransform

logger = logging.getLogger(__name__)


class CityscapesCombineThermalDataset(SegmentationDataset):
    NUM_CLASS = 19
    IGNORE_INDEX = -1
    NAME = "cityscapes_thermal"
    mean = [0.15719692, 0.15214752, 0.15960556]
    std = [0.06431248, 0.06369495, 0.06447389]

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

        self.images = []
        self.mask_paths = []
        self.edge_paths = []
        for split in ["train", "val", "test"]:
            images, mask_paths, edge_paths = FilePathHandler._get_city_pairs(self.root, split)
            self.images.extend(images)
            self.mask_paths.extend(mask_paths)
            self.edge_paths.extend(edge_paths)

        logger.info(f"Found {len(self.images)} images in the folder")

        if len(self.images) != len(self.mask_paths):
            raise ValueError("Mismatch between images and masks")
        if len(self.images) != len(self.edge_paths):
            raise ValueError("Mismatch between images and edges")
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in subfolder of: {root}")

        self.update_normalization(NormalizationTransform(self.mean, self.std))

        valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        key = np.array(
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                1,
                -1,
                -1,
                2,
                3,
                4,
                -1,
                -1,
                -1,
                5,
                -1,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                -1,
                -1,
                16,
                17,
                18,
            ]
        )

        self.update_image_transform(
            CityscapeTransform(
                valid_classes=valid_classes, key=key, ignore_index=self.IGNORE_INDEX
            )
        )

    @property
    def class_names(
        self,
    ):
        return [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicyle",
        ]


if __name__ == "__main__":
    from os import path, sys

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(os.path.join(path.dirname(path.dirname(path.abspath(__file__))), ".."))
    from core.data.samplers import make_data_sampler, make_multiscale_batch_data_sampler
    from torch.utils.data import DataLoader

    from datasets import *

    data_kwargs = {
        "base_size": [960, 328],
        "crop_size": [512, 256],
    }
    train_dataset = CityscapesCombineThermalDataset(
        root="/mnt/ACF29FC2F29F8F68/Work/Deep_Learning/Thermal_Segmentation/Dataset/",
        # root='./../../../../Dataset/',
        split="train",
        mode="train",
        **data_kwargs,
    )

    train_sampler = make_data_sampler(dataset=train_dataset, shuffle=True, distributed=False)

    train_batch_sampler = make_multiscale_batch_data_sampler(
        sampler=train_sampler, batch_size=1, multiscale_step=2, scales=2
    )

    loader_random_sample = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        pin_memory=True,
    )

    x = train_dataset.__getitem__(10)
