import argparse
import logging
from pathlib import Path

import cv2
from seg2edge import seg2edge

# Set up logging
logger = logging.getLogger(__name__)

# path = Path("/mnt/C26EDFBB6EDFA687/lab-work/FTNet/data/processed_dataset/")
# labels_path = ['CITYSCAPE_5000/', 'SODA/', 'SCUTSEG/', 'MFN/']
# radius = [2, 1, 1, 1]

# datasets = ['cityscapes', 'soda', 'scutseg', 'mfn']
# datasets = ["MFN"]


def main(datasets: str, path: str, labels_path: str, radius: list):
    path = Path(path)
    datasets = datasets.split(",")
    labels_path = labels_path.split(",")

    for dataset, data_path, r in zip(datasets, labels_path, radius):
        logger.info(f"Running Edge Detection on {dataset}")
        set_list = (
            ["train"]
            if dataset == "cityscapes"
            else ["train", "val", "test"]
            if dataset == "soda"
            else ["train", "test"]
            if dataset == "scutseg"
            else ["train", "val", "test"]
        )
        for set_item in set_list:
            logger.info(f"Running Set {set_item}")

            save_path = path / data_path / "edges" / set_item
            save_path.mkdir(parents=True, exist_ok=True)
            mask_path = path / data_path / "mask" / set_item
            for file_path in mask_path.iterdir():
                logger.info(f"Filename: {file_path.stem}")
                if file_path.name not in [".", ".."]:
                    mask = cv2.imread(str(file_path))
                    edge_map_bin = seg2edge(mask, r, label_ignore=None, edge_type="regular")
                    cv2.imwrite(str(save_path / file_path.name), edge_map_bin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run edge detection on datasets")
    parser.add_argument(
        "--datasets", type=str, required=True, help="Comma-separated list of datasets"
    )
    parser.add_argument("--path", type=Path, required=True, help="Path to the dataset directory")
    parser.add_argument(
        "--labels-path", type=str, required=True, help="Comma-separated list of labels paths"
    )
    parser.add_argument(
        "--radius", type=int, nargs="+", required=True, help="List of radii for edge detection"
    )

    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

    main(args.datasets, args.path, args.labels_path, args.radius)
