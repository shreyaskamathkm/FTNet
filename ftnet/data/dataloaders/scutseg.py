"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image

from .segbase import SegmentationDataset


class SCUTSEGDataset(SegmentationDataset):
    NUM_CLASS = 10
    IGNORE_INDEX = 255
    NAME = "SCUTSEG"
    BASE_FOLDER = "SCUTSEG"

    def __init__(
        self,
        root="./",
        split="train",
        base_size=1024,
        crop_size=720,
        mode=None,
        logger=None,
        transform=None,
        sobel_edges=False,
    ):
        root = os.path.join(root, self.BASE_FOLDER)
        super().__init__(root, split, mode, base_size, crop_size, logger)
        self.sobel_edges = sobel_edges

        assert os.path.exists(self.root), "Error: data root path is wrong!"

        # FOR same channel
        self.mean = [0.41213047, 0.42389206, 0.416051]
        self.std = [0.13611181, 0.13612076, 0.13611817]

        self.images, self.mask_paths, self.edge_paths = _get_scutseg_pairs(
            self.root, self.split, logger
        )
        assert len(self.images) == len(self.mask_paths)

        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

        self.valid_classes = [0, 7, 24, 25, 26, 27, 13, 21, 28, 17]
        self.class_map = dict(zip(self.valid_classes, range(10)))

    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            index, scale = index
            input_size = self.crop_size[scale]
        else:
            input_size = None

        img = Image.open(self.images[index])
        if self.mode == "test":
            img = self.normalize(img)
            return img, os.path.basename(self.images[index])
            """For Training and validation purposes."""
        mask = Image.open(self.mask_paths[index])

        if self.sobel_edges:
            id255 = np.where(mask == 255)
            no255_gt = np.array(mask)
            no255_gt[id255] = 0
            edge = cv2.Canny(no255_gt, 5, 5, apertureSize=7)
            edge = cv2.dilate(edge, self.edge_kernel)
            edge[edge == 255] = 1
            edge = Image.fromarray(edge)
        else:
            edge = Image.open(self.edge_paths[index])

        # synchrosized transform
        if self.mode == "train":
            img, mask, edge = self._sync_transform(
                img=img, mask=mask, edge=edge, crop_size=input_size
            )
        elif self.mode == "val":
            img, mask, edge = self._val_sync_transform(
                img=img, mask=mask, edge=edge, crop_size=input_size
            )
        else:
            assert self.mode == "testval"
            img, mask, edge = (
                self._img_transform(img),
                self._mask_transform(mask),
                self._edge_transform(edge),
            )

        # general resize, normalize and to Tensor
        img = self.normalize(img)
        return img, mask, edge, os.path.basename(self.images[index])

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def normalize(self, img):
        img = self.im2double(np.array(img))
        img = (img - self.mean) * np.reciprocal(self.std)
        return self.np2Tensor(img).float()

    def _mask_transform(self, mask):
        mask = self.encode_segmap(np.array(mask))
        return torch.LongTensor(mask.astype("int32"))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

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


def _get_scutseg_pairs(folder, split="train", logger=None):
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
                        print(
                            "cannot find the image, mask, or edge:",
                            imgpath,
                            maskpath,
                            edgepath,
                        )
        if logger is not None:
            logger.info(f"Found {len(img_paths)} images in the folder {img_folder}")
        return img_paths, mask_paths, edge_paths

    if split in ("train", "val", "test"):
        img_folder = os.path.join(folder, "image")
        mask_folder = os.path.join(folder, "mask")
        edge_folder = os.path.join(folder, "edges")
        img_paths, mask_paths, edge_paths = get_path_pairs(img_folder, mask_folder, edge_folder)
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
