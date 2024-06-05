"""
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/cityscapes.py
"""

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
