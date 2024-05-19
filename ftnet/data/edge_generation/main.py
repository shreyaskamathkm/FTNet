from pathlib import Path

import cv2
from seg2edge import seg2edge

# datasets = ['cityscapes', 'soda', 'scutseg', 'mfn']
datasets = ["soda"]
path = Path("/mnt/f/lab-work/FTNet/data/processed_dataset/")
labels_path = ["SODA"]
# labels_path = ['CITYSCAPE_5000/', 'SODA/', 'SCUTSEG/', 'MFN/']
radius = [2, 1, 1, 1]

for dataset, data_path, r in zip(datasets, labels_path, radius):
    setList = (
        ["train"]
        if dataset == "cityscapes"
        else ["train", "val", "test"]
        if dataset == "soda"
        else ["train", "test"]
        if dataset == "scutseg"
        else ["train", "val", "test"]
    )
    for set_item in setList:
        save_path = path / data_path / "edges" / set_item
        save_path.mkdir(parents=True, exist_ok=True)
        mask_path = path / data_path / "mask" / set_item
        for file_path in mask_path.iterdir():
            if file_path.name not in [".", ".."]:
                mask = cv2.imread(str(file_path))
                edgeMapBin = seg2edge(mask, r, label_ignore=None, edge_type="regular")
                cv2.imwrite(str(save_path / file_path.name), edgeMapBin)
