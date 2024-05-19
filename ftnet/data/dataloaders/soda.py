import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .segbase import SegmentationDataset

logger = logging.getLogger(__name__)


class SODADataset(SegmentationDataset):
    NUM_CLASS = 21
    IGNORE_INDEX = 255
    NAME = "SODA"
    BASE_FOLDER = "SODA"

    def __init__(
        self,
        root: Path = Path("./datasets/processed_dataset/SODA"),
        split: str = "train",
        base_size: list[int] = 1024,
        crop_size: list[int] = 720,
        mode: str = None,
        sobel_edges: bool = False,
    ):
        root = Path(root) / self.BASE_FOLDER
        super().__init__(root, split, mode, base_size, crop_size)
        assert root.exists(), "Error: data root path is wrong!"
        self.sobel_edges = sobel_edges

        self.mean = [0.41079543, 0.41079543, 0.41079543]
        self.std = [0.18772296, 0.18772296, 0.18772296]

        self.images, self.mask_paths, self.edge_paths = _get_soda_pairs(root, self.split)
        assert len(self.images) == len(self.mask_paths), "Mismatch between images and masks"
        assert len(self.images) == len(self.edge_paths), "Mismatch between images and edges"

        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in subfolder of: {root}")

    def __getitem__(self, index):
        scale = None
        if isinstance(index, (list, tuple)):
            index, scale = index

        img = Image.open(self.images[index]).convert("RGB")

        if self.mode == "test":
            img = self.normalize(img)
            return img, self.images[index].name

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

        # synchronized transform
        if self.mode == "train":
            img, mask, edge = self._sync_transform(img=img, mask=mask, edge=edge, scale=scale)
        elif self.mode == "val":
            img, mask, edge = self._val_sync_transform(img=img, mask=mask, edge=edge, scale=scale)
        else:
            assert self.mode == "testval"
            img, mask, edge = (
                self._img_transform(img),
                self._mask_transform(mask),
                self._edge_transform(edge),
            )

        # general resize, normalize and to Tensor
        img = self.normalize(img)
        return img, mask, edge, self.images[index].name

    def normalize(self, img):
        img = self.im2double(np.array(img))
        img = (img - self.mean) * np.reciprocal(self.std)
        img = self.np2Tensor(img).float()
        return img

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype("int32"))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def class_names(self):
        return [
            "background",
            "person",
            "building",
            "tree",
            "road",
            "pole",
            "grass",
            "door",
            "table",
            "chair",
            "car",
            "bicycle",
            "lamp",
            "monitor",
            "trafficCone",
            "trash can",
            "animal",
            "fence",
            "sky",
            "river",
            "sidewalk",
        ]


def _get_soda_pairs(folder: Path, split: str = "train"):
    def get_path_pairs(img_folder: Path, mask_folder: Path, edge_folder: Path):
        img_folder = img_folder / split
        mask_folder = mask_folder / split
        edge_folder = edge_folder / split
        img_paths = []
        mask_paths = []
        edge_paths = []

        for imgpath in img_folder.rglob("*.jpg"):
            maskname = f"{imgpath.stem}.png"
            maskpath = mask_folder / maskname
            edgepath = edge_folder / maskname

            if maskpath.is_file() and edgepath.is_file():
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
                edge_paths.append(edgepath)
            else:
                logger.warning("Cannot find the {imgpath}, {maskpath}, or {edgepath}")

        logger.info(f"Found {len(img_paths)} images in the folder {img_folder}")
        return img_paths, mask_paths, edge_paths

    if split in {"train", "val", "test"}:
        img_folder = folder / "image"
        mask_folder = folder / "mask"
        edge_folder = folder / "edges"
        img_paths, mask_paths, edge_paths = get_path_pairs(img_folder, mask_folder, edge_folder)
        return img_paths, mask_paths, edge_paths


if __name__ == "__main__":
    from pathlib import Path

    from core.data.samplers import make_data_sampler, make_multiscale_batch_data_sampler
    from torch.utils.data import DataLoader

    from datasets import *

    data_kwargs = {"base_size": [[520, 520]], "crop_size": [[480, 480]]}

    train_dataset = SODADataset(
        root="/mnt/f/lab-work/FTNet/data/processed_dataset/",
        split="test",
        mode="train",
        **data_kwargs,
    )

    train_sampler = make_data_sampler(dataset=train_dataset, shuffle=True, distributed=False)

    train_batch_sampler = make_multiscale_batch_data_sampler(
        sampler=train_sampler, batch_size=1, multiscale_step=1, scales=1
    )

    loader_random_sample = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        pin_memory=True,
    )

    x = train_dataset.__getitem__(10)
