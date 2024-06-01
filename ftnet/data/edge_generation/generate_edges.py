"""
Example run:   python -m ftnet.data.edge_generation.generate_edges   --datasets <dataset name to generate edges>  --save-path <path to save dataset> --radius <corresponding radii>
python -m ftnet.data.edge_generation.generate_edges  --datasets cityscape,soda,scutseg,mfn --save-path ./data/processed_dataset/ --radius 2,1,1,1
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from rich.logging import RichHandler

from .seg2edge import seg2edge

logger = logging.getLogger(__name__)


# Set up logging
# path = Path("/mnt/C26EDFBB6EDFA687/lab-work/FTNet/data/processed_dataset/")
# labels_path = ['CITYSCAPE_5000/', 'SODA/', 'SCUTSEG/', 'MFN/']
# radius = [2, 1, 1, 1]

# datasets = ['cityscapes', 'soda', 'scutseg', 'mfn']
# datasets = ["MFN"]


DATASET_SETS = {
    "cityscape": ["train", "val", "test"],
    "soda": ["train", "test"],
    "scutseg": ["train", "val"],
    "mfn": ["train", "val", "test"],
}


def process_set(dataset, set_item, path, r):
    save_path = path / dataset / "edges" / set_item
    save_path.mkdir(parents=True, exist_ok=True)
    mask_path = path / dataset / "mask" / set_item
    for file_path in mask_path.iterdir():
        logger.info(f"Filename: {file_path.stem}")
        if file_path.name not in [".", ".."]:
            mask = cv2.imread(str(file_path))
            edge_map_bin = seg2edge(mask, int(r), label_ignore=None, edge_type="regular")
            cv2.imwrite(str(save_path / file_path.name), edge_map_bin)


def main(datasets: str, path: str, radius: list, num_workers: int):
    datasets = datasets.split(",")
    radius = radius.split(",")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for dataset, r in zip(datasets, radius):
            logger.info(f"Running Edge Detection on {dataset}")
            set_list = DATASET_SETS[dataset]

            for set_item in set_list:
                logger.info(f"Running Set {set_item}")
                futures.append(executor.submit(process_set, dataset, set_item, Path(path), r))

        # Wait for all tasks to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[RichHandler()],
    )

    parser = argparse.ArgumentParser(description="Run edge detection on datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        default="cityscape,soda,scutseg,mfn",
        help="Comma-separated list of datasets, example: 'cityscape,soda,scutseg,mfn'",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default="./data/processed_dataset/",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--radius",
        type=str,
        default="2,1,1,1",
        help="List of radii for edge detection example: '2,1,1,1' i.e. for cityscape,soda,scutseg,mfn respectively",
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Number of workers to use to generate edges"
    )
    args = parser.parse_args()

    main(args.datasets, args.save_path, args.radius, args.num_workers)
