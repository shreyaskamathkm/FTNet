from .dataloaders import (
    CityscapesCombineThermalDataset,
    CityscapesThermalsSplitDataset,
    MFNDataset,
    SCUTSEGDataset,
    SODADataset,
)

__all__ = ["datasets", "get_segmentation_dataset"]


datasets = {
    "cityscapes_thermal_combine": CityscapesCombineThermalDataset,
    "cityscapes_thermal_split": CityscapesThermalsSplitDataset,
    "soda": SODADataset,
    "mfn": MFNDataset,
    "scutseg": SCUTSEGDataset,
}


def get_segmentation_dataset(name: str, **kwargs: str):
    """Segmentation Datasets."""
    return datasets[name.lower()](**kwargs)
