from .dataloaders import (
    CityscapesCombineThermalDataset,
    CityscapesThermalsSplitDataset,
    SODADataset,
    MFNDataset,
    SCUTSEGDataset,
)

datasets = {
    "cityscapes_thermal_combine": CityscapesCombineThermalDataset,
    "cityscapes_thermal_split": CityscapesThermalsSplitDataset,
    "soda": SODADataset,
    "mfn": MFNDataset,
    "scutseg": SCUTSEGDataset,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets."""
    return datasets[name.lower()](**kwargs)