from .cityscapes_thermal import CityscapesCombineThermalDataset
from .cityscapes_thermal_split import CityscapesThermalSplitDataset
from .load_images import LoadImages
from .mfn import MFNDataset
from .scutseg import SCUTSEGDataset
from .soda import SODADataset

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
