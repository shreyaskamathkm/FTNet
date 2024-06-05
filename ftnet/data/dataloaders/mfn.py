"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

import logging
from pathlib import Path
from typing import List

from .base_dataloader import SegmentationDataset
from .file_path_handler import FilePathHandler
from .transforms import NormalizationTransform

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class MFNDataset(SegmentationDataset):
    NUM_CLASS = 9
    IGNORE_INDEX = 255
    NAME = "mfn"
    mean = [0.3954101, 0.3954101, 0.3954101]
    std = [0.07577764, 0.07577764, 0.07577764]

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

        self.images, self.mask_paths, self.edge_paths = FilePathHandler._get_pairs(
            self.root, self.split
        )

        if len(self.images) != len(self.mask_paths):
            raise ValueError("Mismatch between images and masks")
        if len(self.images) != len(self.edge_paths):
            raise ValueError("Mismatch between images and edges")
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in subfolder of: {root}")

        self.update_normalization(NormalizationTransform(self.mean, self.std))

    @property
    def class_names(self) -> List[str]:
        return [
            "unlabeled",
            "car",
            "person",
            "bike",
            "curve",
            "car_stop",
            "guardrail",
            "color_cone",
            "bump",
        ]
