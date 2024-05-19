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


class CityscapesThermalsSplitDataset(SegmentationDataset):
    NUM_CLASS = 19
    IGNORE_INDEX = -1
    NAME = "cityscapes"
    BASE_FOLDER = "Cityscapes/data_proc/"

    def __init__(
        self,
        root="./datasets/Cityscapes",
        split="train",
        base_size=1024,
        crop_size=720,
        mode=None,
        logger=None,
        sobel_edges=False,
        transform=None,
    ):
        root = os.path.join(root, self.BASE_FOLDER)
        super(CityscapesThermalsSplitDataset, self).__init__(
            root, split, mode, base_size, crop_size, logger
        )
        self.sobel_edges = sobel_edges
        assert os.path.exists(self.root), "Error: data root path is wrong!"

        self.images, self.mask_paths, self.edge_paths = _get_city_pairs(
            self.root, self.split, logger
        )
        assert len(self.images) == len(self.mask_paths) == len(self.edge_paths)

        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [
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

        self._key = np.array(
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
        # [-1, ..., 33]
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype("int32")
        self.mean = [0.15719692, 0.15214752, 0.15960556]
        self.std = [0.06431248, 0.06369495, 0.06447389]

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert value in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        if isinstance(index, list) or isinstance(index, tuple):
            index, scale = index
            input_size = self.crop_size[scale]
        else:
            input_size = None

        img = Image.open(self.images[index]).convert("RGB")

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
            img, mask = self._img_transform(img), self._mask_transform(mask)

        # general resize, normalize and to Tensor
        img = self.normalize(img)
        return img, mask, edge, os.path.basename(self.images[index])

    def normalize(self, img):
        img = self.im2double(np.array(img))
        img = (img - self.mean) * np.reciprocal(self.std)
        img = self.np2Tensor(img).float()
        return img

    def _edge_transform(self, edge):
        edge = np.array(edge)
        edge = np.sum(edge, 2)
        edge[edge > 0] = 1
        edge[edge <= 0] = 0
        return torch.LongTensor(edge.astype("int32"))

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype("int32"))
        target[target == -1] = self.IGNORE_INDEX
        return torch.LongTensor(np.array(target).astype("int32"))

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


def _get_city_pairs(folder, split="train", logger=None):
    def get_path_pairs(img_folder, mask_folder, edge_folder):
        img_paths = []
        mask_paths = []
        edge_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".jpg"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))

                    maskname = filename.replace(
                        "leftImg8bit_synthesized_image.jpg", "gtFine_labelIds.png"
                    )
                    maskpath = os.path.join(mask_folder, foldername, maskname)

                    edgename = filename.replace(
                        "leftImg8bit_synthesized_image.jpg", "gtFine_edge.png"
                    )
                    edgepath = os.path.join(mask_folder, foldername, edgename)

                    if (
                        os.path.isfile(imgpath)
                        and os.path.isfile(maskpath)
                        and os.path.isfile(edgepath)
                    ):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                        edge_paths.append(edgepath)
                    else:
                        print("cannot find the mask or image:", imgpath, maskpath)
        if logger is not None:
            logger.info(f"Found {len(img_paths)} images in the folder {img_folder}")
        return img_paths, mask_paths, edge_paths

    if split in ("train", "val"):
        # "./Cityscapes/leftImg8bit/train" or "./Cityscapes/leftImg8bit/val"
        img_folder = os.path.join(folder, "TIR_leftImg8bit/" + split)
        # "./Cityscapes/gtFine/train" or "./Cityscapes/gtFine/val"
        mask_folder = os.path.join(folder, "gtFine/" + split)
        edge_folder = os.path.join(folder, "gtFine/" + split)
        # img_paths与mask_paths的顺序是一一对应的
        img_paths, mask_paths, edge_paths = get_path_pairs(
            img_folder, mask_folder, edge_folder
        )
        return img_paths, mask_paths, edge_paths
    else:
        assert split == "trainval"
        print("trainval set")
        train_img_folder = os.path.join(folder, "leftImg8bit/train")
        train_mask_folder = os.path.join(folder, "gtFine/train")
        val_img_folder = os.path.join(folder, "leftImg8bit/val")
        val_mask_folder = os.path.join(folder, "gtFine/val")
        train_img_paths, train_mask_paths = get_path_pairs(
            train_img_folder, train_mask_folder
        )
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == "__main__":
    from os import path, sys

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from core.data.samplers import make_data_sampler, make_multiscale_batch_data_sampler
    from torch.utils.data import DataLoader

    from datasets import *

    data_kwargs = {"base_size": [960, 328], "crop_size": [512, 256]}

    train_dataset = CityscapesThermalsSplitDataset(
        root="/mnt/ACF29FC2F29F8F68/Work/Deep_Learning/Thermal_Segmentation/Dataset/",
        split="train",
        mode="train",
        **data_kwargs,
    )

    train_sampler = make_data_sampler(
        dataset=train_dataset, shuffle=False, distributed=False
    )

    train_batch_sampler = make_multiscale_batch_data_sampler(
        sampler=train_sampler, batch_size=10, multiscale_step=1, scales=2
    )

    loader_random_sample = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=4,
        pin_memory=True,
    )

    x = train_dataset.__getitem__(100)
