"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

import logging
from pathlib import Path
from typing import List

from .base_dataloader import SegmentationDataset
from .cityscapes_thermal import CityscapesCombineThermalDataset
from .mfn import MFNDataset
from .scutseg import SCUTSEGDataset
from .soda import SODADataset
from .transforms import NormalizationTransform

logger = logging.getLogger(__name__)

# Define the base class mappings
dataset_classes = {
    "cityscapes": CityscapesCombineThermalDataset,
    "soda": SODADataset,
    "mfn": MFNDataset,
    "scutseg": SCUTSEGDataset,
}


class LoadImages(SegmentationDataset):
    """Dataset class for loading images for inference or evaluation."""

    def __init__(
        self,
        root: Path = "./Dataset/",
        dataset: str = "soda",
        mode: str = "infer",
        sobel_edges: bool = False,
        base_size: List[List[int]] = [[520, 520]],
    ) -> None:
        """Initialize the dataset.

        Args:
            root (Path): Root folder containing the dataset. Defaults to "./Dataset/".
            dataset (str): Name of the dataset. Defaults to "soda".
            mode (str): Mode of the dataset ('train', 'val', 'test', 'infer'). Defaults to "infer".
            sobel_edges (bool, optional): Whether to apply Sobel edge detection. Defaults to False.

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        super().__init__(root, None, mode, base_size, None, sobel_edges)

        assert root.exists(), "Error: data root path is wrong!"

        base_class = dataset_classes.get(dataset)

        if not base_class:
            raise ValueError(f"Dataset name '{dataset}' is not recognized")

        self.images = list(self.root.glob("**/*[.jpg,png]"))
        logger.info(f"Found {len(self.images)} images in the folder {self.root}")
        self.update_normalization(NormalizationTransform(base_class.mean, base_class.std))
