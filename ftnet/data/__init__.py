from .dataloaders.cityscapes_thermal import CityscapesCombineThermalDataset
from .dataloaders.cityscapes_thermal_split import CityscapesThermalSplitDataset
from .dataloaders.load_images import LoadImages
from .dataloaders.mfn import MFNDataset
from .dataloaders.scutseg import SCUTSEGDataset
from .dataloaders.soda import SODADataset

__all__ = ["get_segmentation_dataset"]


datasets = {
    "cityscapes_thermal_combine": CityscapesCombineThermalDataset,
    "cityscapes_thermal_split": CityscapesThermalSplitDataset,
    "soda": SODADataset,
    "mfn": MFNDataset,
    "scutseg": SCUTSEGDataset,
    "load_image": LoadImages,
}


def get_segmentation_dataset(name: str, **kwargs: str):
    """Segmentation Datasets."""
    return datasets[name.lower()](**kwargs)
