"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

import logging
from pathlib import Path
from typing import List

import numpy as np

from .base_dataloader import SegmentationDataset
from .file_path_handler import FilePathHandler
from .transforms import CityscapeTransform, NormalizationTransform

logger = logging.getLogger(__name__)


class CityscapesCombineThermalDataset(SegmentationDataset):
    """Cityscapes Combine Thermal Dataset for segmentation tasks.

    Attributes:
        NUM_CLASS (int): Number of classes.
        IGNORE_INDEX (int): Index to ignore in the target.
        NAME (str): Name of the dataset.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
    """

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
        """Initialize the dataset.

        Args:
            root (Path): Root directory of the dataset.
            split (str, optional): Split of the dataset ('train', 'val', 'test'). Defaults to 'train'.
            base_size (List[List[int]], optional): Base size for resizing. Defaults to [[520, 520]].
            crop_size (List[List[int]], optional): Crop size for cropping. Defaults to [[480, 480]].
            mode (str, optional): Mode of the dataset. Defaults to None.
            sobel_edges (bool, optional): Whether to use Sobel edges. Defaults to False.
        """
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
    def class_names(self) -> List[str]:
        """Get the class names for the dataset.

        Returns:
            List[str]: List of class names.
        """
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
            "bicycle",
        ]
