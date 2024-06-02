"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/segbase.py
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .data_attributes import ImageSize, ImageSizes
from .transforms import ImageTransform, NormalizationTransform, ResizingTransformations

logger = logging.getLogger(__name__)
__all__ = ["SegmentationDataset"]


class SegmentationDataset:
    """Segmentation Base Dataset.

    Args:
        root (Path): Path to the dataset root directory.
        split (str): Dataset split, e.g., 'train', 'val', 'test'.
        mode (str): Mode for the dataset, e.g., 'train', 'val', 'testval'.
        base_size (List[int]): Base size of the image, default is [520].
        crop_size (List[int]): Crop size of the image, default is [480].
    """

    def __init__(
        self,
        root: Path,
        split: str,
        mode: str,
        base_size: List[List[int]] = [[520, 520]],
        crop_size: List[List[int]] = [[480, 480]],
        sobel_edges: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.mode = mode or split
        self.base_size = base_size
        self.crop_size = crop_size

        if not root.exists():
            raise FileNotFoundError(f"Error: data root path {root} is wrong!")

        self.sobel_edges = sobel_edges

        if self.sobel_edges:
            edge_radius = 7
            self.edge_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (edge_radius, edge_radius)
            )

        if self.mode != "test":
            base_sizes = [ImageSize(base[0], base[1]) for base in base_size]
            crop_sizes = [ImageSize(crop[0], crop[1]) for crop in crop_size]
            self.sizes = [
                ImageSizes(base_size=base, crop_size=crop)
                for crop, base in zip(crop_sizes, base_sizes)
            ]

        self.normalization = None
        self.transform = ResizingTransformations(self.sizes)
        self.image_transform = ImageTransform

    def __getitem__(self, index: Union[int, List, Tuple]):
        scale = None
        if isinstance(index, (list, tuple)):
            index, scale = index

        if self.NAME == "mfn":
            img = np.asarray(Image.open(self.images[index]))[:, :, 3]  # type: ignore
            img = Image.fromarray(img).convert("RGB")
        else:
            img = Image.open(self.images[index]).convert("RGB")

        if self.mode == "infer":
            img = self.image_transform.img_transform(img)
            img = self.normalization.normalize(img)
            return img, self.images[index].name

        mask = Image.open(self.mask_paths[index])

        edge = None
        if not self.sobel_edges:
            edge = Image.open(self.edge_paths[index])

        if not edge:
            id255 = np.where(mask == 255)
            no255_gt = np.array(mask)
            no255_gt[id255] = 0
            edge = cv2.Canny(no255_gt, 5, 5, apertureSize=7)
            edge = cv2.dilate(edge, self.edge_kernel)
            edge[edge == 255] = 1
            edge = Image.fromarray(edge)

        if self.mode == "train":
            img, mask, edge = self.transform.sync_transform(img, mask, edge, scale)
        elif self.mode == "val":
            img, mask, edge = self.transform.val_sync_transform(img, mask, edge)

        img = self.image_transform.img_transform(img)
        mask = self.image_transform.mask_transform(mask)
        edge = self.image_transform.edge_transform(edge)

        img = self.normalization.normalize(img)
        return img, mask, edge, self.images[index].name

    @property
    def num_class(self) -> int:
        """Number of categories."""
        raise NotImplementedError

    @property
    def pred_offset(self) -> int:
        """Prediction offset."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.images)

    def update_normalization(self, normalization: NormalizationTransform):
        self.normalization = normalization

    def update_image_transform(self, transform: ImageTransform):
        self.image_transform = transform
