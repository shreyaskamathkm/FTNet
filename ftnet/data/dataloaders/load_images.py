"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

import os

from .base_dataloader import SegmentationDataset
from .cityscapes_thermal import CityscapesCombineThermalDataset
from .mfn import MFNDataset
from .scutseg import SCUTSEGDataset
from .soda import SODADataset
from .transforms import NormalizationTransform

# Define the base class mappings
dataset_classes = {
    "cityscapes": CityscapesCombineThermalDataset,
    "soda": SODADataset,
    "mfn": MFNDataset,
    "scutseg": SCUTSEGDataset,
}


class LoadImages(SegmentationDataset):
    def __init__(
        self,
        root="./Dataset/",
        dataset="soda",
        mode="infer",
        sobel_edges=False,
    ):
        super().__init__(root, None, mode, None, None, sobel_edges)

        root = root
        assert root.exists(), "Error: data root path is wrong!"

        base_class = dataset_classes.get(dataset)

        if not base_class:
            raise ValueError(f"Dataset name '{dataset}' is not recognized")

        self.images = list(self.root.glob("**/*[.jpg,png]"))
        self.update_normalization(NormalizationTransform(base_class.mean, base_class.std))


if __name__ == "__main__":
    from os import path, sys

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(os.path.join(path.dirname(path.dirname(path.abspath(__file__))), ".."))
    from core.data.samplers import make_data_sampler, make_multiscale_batch_data_sampler
    from datasets import *
    from torch.utils.data import DataLoader

    data_kwargs = {"base_size": [520], "crop_size": [480]}

    train_dataset = MFNDataset(
        root="/mnt/ACF29FC2F29F8F68/Work/Deep_Learning/Thermal_Segmentation/Dataset/",
        split="test",
        mode="testval",
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

    x = train_dataset.__getitem__(200)
