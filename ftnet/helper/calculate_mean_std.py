"""
Example Usage: python -m ftnet.data.calculate_mean_std  --train_folder <path to train data folder>
"""

import argparse
import time
from multiprocessing import Pool as ThreadPool
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tqdm


def im2double(im: np.ndarray) -> np.ndarray:
    """Convert an image to double precision (float) and normalize to the range
    [0, 1].

    Args:
        im (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized image array.
    """
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return (
        im.astype(float) / info.max
    )  # Divide all values by the largest possible value in the datatype


def is_image_file(filename: Path) -> bool:
    """Check if a file is an image based on its extension.

    Args:
        filename (Path): Path to the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return filename.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def calculate_MSTD(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the mean and standard deviation of an image.

    Args:
        path (Path): Path to the image file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and standard deviation of the image.
    """
    img = cv2.imread(str(path), 1)
    img = im2double(img)
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    return mean, std


def main(test_folder: Path) -> None:
    """Main function to calculate the mean and standard deviation for all
    images in a folder.

    Args:
        test_folder (Path): Path to the folder containing the images.
    """
    image_list = [x for x in test_folder.glob("*") if is_image_file(x)]

    start = time.time()
    with ThreadPool(6) as p:
        patch_grids = list(tqdm.tqdm(p.imap(calculate_MSTD, image_list), total=len(image_list)))

    all_img_means = [x[0] for x in patch_grids]
    all_img_std = [x[1] for x in patch_grids]

    mean_per_channel = np.mean(np.array(all_img_means), axis=0)
    std_per_channel = np.mean(np.array(all_img_std), axis=0)

    print(f"Total Time = {time.time() - start}")
    print(mean_per_channel)
    print(std_per_channel)

    np.savez("mean_std.npz", mean=mean_per_channel, std=std_per_channel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finding Mean and Std Deviation")
    parser.add_argument(
        "--train_folder",
        type=Path,
        default="./data/processed_dataset/cityscape/image/train/",
        help="Dataset folder for calculating the mean and standard deviation.",
    )
    args = parser.parse_args()
    main(args.train_folder)
