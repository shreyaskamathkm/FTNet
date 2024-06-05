from .dataloaders import (
    CityscapesCombineThermalDataset,
    CityscapesThermalSplitDataset,
    LoadImages,
    MFNDataset,
    SCUTSEGDataset,
    SODADataset,
)

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
