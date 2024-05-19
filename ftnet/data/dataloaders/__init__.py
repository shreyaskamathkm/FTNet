# try:
from .cityscapes_thermal import CityscapesCombineThermalDataset
from .cityscapes_thermal_split import CityscapesThermalsSplitDataset
from .mfn import MFNDataset
from .scutseg import SCUTSEGDataset
from .soda import SODADataset


# except:
#     from datasets.dataloaders.cityscapes_thermal import CityscapesCombineThermalDataset
#     from datasets.dataloaders.soda import SODADataset
#     from datasets.dataloaders.mfn import MFNDataset
#     from datasets.dataloaders.cityscapes_thermal_split import CityscapesThermalsSplitDataset
#     from datasets.dataloaders.scutseg import SCUTSEGDataset
