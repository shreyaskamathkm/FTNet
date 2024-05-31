"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .segbase import SegmentationDataset

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

        self.sobel_edges = sobel_edges

        assert os.path.exists(self.root), "Error: data root path is wrong!"

        # FOR same channel

        self.images, self.mask_paths, self.edge_paths = SCUTSEGDataset._get_scutseg_pairs(
            self.root, self.split
        )
        if len(self.images) != len(self.mask_paths):
            raise ValueError("Mismatch between images and masks")
        if len(self.images) != len(self.edge_paths):
            raise ValueError("Mismatch between images and edges")
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in subfolder of: {root}")

        self.valid_classes = [0, 7, 24, 25, 26, 27, 13, 21, 28, 17]
        self.class_map = dict(zip(self.valid_classes, range(10)))

    def __getitem__(self, index: Union[int, List, Tuple]):
        scale = None
        if isinstance(index, (list, tuple)):
            index, scale = index

        # reading the images
        img = Image.open(self.images[index]).convert("RGB")

        if self.mode == "infer":
            img = self.normalize(img)
            return img, self.images[index].name

        # reading the mask
        mask = Image.open(self.mask_paths[index])

        # testing if sobel edge filter is required
        edge = None
        if not self.sobel_edges:
            edge = Image.open(self.edge_paths[index])

        # post process as required
        img, mask, edge = self.post_process(img, mask, edge, scale)

        return img, mask, edge, self.images[index].name

    def _encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def _mask_transform(self, mask):
        mask = self._encode_segmap(np.array(mask))
        return torch.LongTensor(mask.astype("int32"))

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

    @staticmethod
    def _get_scutseg_pairs(folder, split="train"):
        def get_path_pairs(img_folder, mask_folder, edge_folder):
            img_folder = os.path.join(img_folder, split)
            mask_folder = os.path.join(mask_folder, split)
            edge_folder = os.path.join(edge_folder, split)
            img_paths = []
            mask_paths = []
            edge_paths = []
            for root, _, files in os.walk(img_folder):
                for filename in files:
                    if filename.endswith(".jpg"):
                        imgpath = os.path.join(root, filename)
                        maskname = filename.replace(".jpg", "_labelIds.png")
                        maskpath = os.path.join(mask_folder, maskname)
                        edgepath = os.path.join(edge_folder, maskname)

                        if (
                            os.path.isfile(imgpath)
                            and os.path.isfile(maskpath)
                            and os.path.isfile(edgepath)
                        ):
                            img_paths.append(imgpath)
                            mask_paths.append(maskpath)
                            edge_paths.append(edgepath)
                        else:
                            logger.warning(f"Cannot find the {imgpath}, {maskpath}, or {edgepath}")

            logger.info(f"Found {len(img_paths)} images in the folder {img_folder}")

            return img_paths, mask_paths, edge_paths

        if split in ("train", "val", "test"):
            img_folder = os.path.join(folder, "image")
            mask_folder = os.path.join(folder, "mask")
            edge_folder = os.path.join(folder, "edges")
            img_paths, mask_paths, edge_paths = get_path_pairs(
                img_folder, mask_folder, edge_folder
            )
            return img_paths, mask_paths, edge_paths

        raise ValueError("Split type unknown")


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
