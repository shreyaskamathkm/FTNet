"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/segbase.py
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)
__all__ = ["SegmentationDataset"]


@dataclass
class ImageSize:
    """Class for storing image dimensions."""

    width: int
    height: int


@dataclass
class ImageSizes:
    """Class for storing base and crop sizes of images."""

    base_size: ImageSize
    crop_size: ImageSize


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
    ) -> None:
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode or split
        self.base_size = base_size
        self.crop_size = crop_size

        edge_radius = 7
        self.edge_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (edge_radius, edge_radius)
        )

        if self.mode != "testval":
            base_sizes = [ImageSize(base[0], base[1]) for base in base_size]
            crop_sizes = [ImageSize(crop[0], crop[1]) for crop in crop_size]
            self.sizes = [
                ImageSizes(base_size=base, crop_size=crop)
                for crop, base in zip(crop_sizes, base_sizes)
            ]

    def _val_sync_transform(
        self,
        img: Image.Image,
        mask: Image.Image,
        edge: Image.Image,
        scale: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synchronously transform validation images."""
        outsize = self.crop_size[0]
        short_size = outsize
        w, h = img.size

        if w > h:
            oh = short_size
            ow = int(w * oh / h)
        else:
            ow = short_size
            oh = int(h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.0))
        y1 = int(round((h - outsize) / 2.0))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        edge = edge.crop((x1, y1, x1 + outsize, y1 + outsize))

        # final transform
        img = self._img_transform(img)
        mask = self._mask_transform(mask)
        edge = self._edge_transform(edge)

        return img, mask, edge

    def _sync_transform(
        self,
        img: Image.Image,
        mask: Image.Image,
        edge: Image.Image,
        scale: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synchronously transform training images with random
        augmentations."""
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

        if scale is not None:
            crop_size = self.sizes[scale].crop_size
            base_size = self.sizes[scale].base_size
        else:
            crop_size = self.sizes[0].crop_size
            base_size = self.sizes[0].base_size

        # random scale (short edge)
        short_size = random.randint(
            int(min(base_size.width, base_size.height) * 0.5),
            int(min(base_size.width, base_size.height) * 2.0),
        )
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(h * ow / w)
        else:
            oh = short_size
            ow = int(w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < max(crop_size.width, crop_size.height):
            padh = crop_size.height - oh if oh < crop_size.height else 0
            padw = crop_size.width - ow if ow < crop_size.width else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=0)

        # random crop
        w, h = img.size
        x1 = random.randint(0, w - crop_size.width)
        y1 = random.randint(0, h - crop_size.height)
        img = img.crop((x1, y1, x1 + crop_size.width, y1 + crop_size.height))
        mask = mask.crop((x1, y1, x1 + crop_size.width, y1 + crop_size.height))
        edge = edge.crop((x1, y1, x1 + crop_size.width, y1 + crop_size.height))

        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        # final transform
        img = self._img_transform(img)
        mask = self._mask_transform(mask)
        edge = self._edge_transform(edge)

        return img, mask, edge

    def im2double(self, im_: np.ndarray) -> np.ndarray:
        """Convert image to double precision."""
        info = np.iinfo(im_.dtype)  # Get the data type of the input image
        return (
            im_.astype(float) / info.max
        )  # Divide all values by the largest possible value in the datatype

    def np2Tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        if len(array.shape) == 3:
            np_transpose = np.ascontiguousarray(array.transpose((2, 0, 1)))
        else:
            np_transpose = np.ascontiguousarray(array[np.newaxis, :])
        return torch.from_numpy(np_transpose).float()

    def _img_transform(self, img: Image.Image) -> np.ndarray:
        """Transform image to numpy array."""
        return np.array(img)

    def _mask_transform(self, mask: Image.Image) -> np.ndarray:
        """Transform mask to numpy array."""
        return np.array(mask).astype("int32")

    def _edge_transform(self, edge: Image.Image) -> np.ndarray:
        """Transform edge to numpy array."""
        return np.array(edge).astype("int32")

    @property
    def num_class(self) -> int:
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self) -> int:
        """Prediction offset."""
        return 0
