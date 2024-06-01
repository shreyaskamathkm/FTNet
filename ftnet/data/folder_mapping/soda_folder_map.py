"""
Example run:   python -m ftnet.folder_mapping.utils.soda_folder_map   --input-image-path <path to original dataset>  --save-path <path to save dataset>
python -m ftnet.folder_mapping.utils.soda_folder_map  --input-image-path ./data/original/SODA/InfraredSemanticLabel/ --save-path ./data/processed_dataset/
"""

import argparse
import logging
import shutil
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.logging import RichHandler
from scipy.stats import wasserstein_distance

# Set up logging
logger = logging.getLogger(__name__)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def plot_hist(split1_hist, split2_hist, num_per_class_perc, num_labels=21):
    # Some visualizations
    # Bar width
    bar_width = 0.35
    categories = np.arange(1, num_labels + 1)
    percentile_values = np.arange(10, 101, 10)  # [25, 50, 75, 100]
    percentile = np.percentile(num_per_class_perc, percentile_values)
    for i, p in enumerate(percentile):
        if i == 0:
            split1_hist_ = split1_hist[split1_hist <= p]
            split2_hist_ = split2_hist[split1_hist <= p]
            categories_ = categories[split1_hist <= p]
        else:
            split1_hist_ = split1_hist[(split1_hist <= p) & (split1_hist > percentile[i - 1])]
            split2_hist_ = split2_hist[(split1_hist <= p) & (split1_hist > percentile[i - 1])]
            categories_ = categories[(split1_hist <= p) & (split1_hist > percentile[i - 1])]
        # Set up figure and axis
        _, ax = plt.subplots()
        # Bar plot for the first vector
        ax.bar(categories_, split1_hist_, label="Split 1")
        # Bar plot for the second vector
        ax.bar(categories_ + bar_width, split2_hist_, label="Split 2")
        # Add labels, title, and legend
        ax.set_xlabel("Class")
        ax.set_ylabel("Percentage")
        ax.set_title(f"Distribution of Quantil {percentile_values[i]}")
        ax.legend()
        # Show the plot
        plt.savefig(f"./split_hist_{i}.png")


def stratify(file_path, num_permutations=50000, split_perc=0.5, random_seed=42, num_labels=21):
    """
    Adapted from https://stackoverflow.com/questions/74202417/stratified-sampling-for-semantic-segmentation
    """

    # guarantee that all classes are in both splits
    min_perc = split_perc / 10

    # Get labels from label files and create histograms
    label_histograms = []
    for path in file_path:
        labels = cv2.imread(path)
        histogram, _ = np.histogram(labels, bins=np.arange(1, num_labels + 2))
        label_histograms.append(histogram)
    # calculate overall histogram and some statistics
    label_histograms = np.array(label_histograms)
    num_per_class = np.sum(label_histograms, axis=0)
    num_per_class_perc = num_per_class / np.sum(num_per_class)
    min_per_class = num_per_class * min_perc

    # initialize permutation
    idx = np.arange(len(file_path))
    permutations = []
    rng = np.random.default_rng(seed=random_seed)
    for _ in range(num_permutations):
        permuted_vector = rng.permutation(idx)
        permutations.append(permuted_vector)

    # performs trials and calculate Earth mover distance
    num_test_perc = int(len(file_path) * split_perc)
    emd_distance = []
    for perm in permutations:
        # test data
        split1_idx = perm[:num_test_perc]
        split1_hist = np.sum(label_histograms[split1_idx], axis=0)
        if np.any(split1_hist < min_per_class):
            emd_distance.append(np.nan)
            continue
        split1_hist = split1_hist / np.sum(split1_hist)
        # train data
        split2_idx = perm[num_test_perc:]
        split2_hist = np.sum(label_histograms[split2_idx], axis=0)
        if np.any(split2_hist < min_per_class):
            emd_distance.append(np.nan)
            continue
        split2_hist = split2_hist / np.sum(split2_hist)
        # add Earth Mover Distance
        emd_distance.append(wasserstein_distance(split1_hist, split2_hist))

    # find best split (smallest EMD value)
    emd_distance = np.asarray(emd_distance)
    emd_distance_min_idx = np.nanargmin(emd_distance)
    perm_final = permutations[emd_distance_min_idx]
    # split1 data
    split1_idx = perm_final[:num_test_perc]
    split1_hist = np.sum(label_histograms[split1_idx], axis=0)
    split1_hist = split1_hist / np.sum(split1_hist)
    # split2 data
    split2_idx = perm_final[num_test_perc:]
    split2_hist = np.sum(label_histograms[split2_idx], axis=0)
    split2_hist = split2_hist / np.sum(split2_hist)
    plot_hist(split1_hist, split2_hist, num_per_class_perc, num_labels=num_labels)

    return split1_idx, split2_idx


# Copying files to different folders
def copying(tiles, path_label, basepath, fileset_path):
    tiles.set_index(path_label, inplace=True)
    for img_path in tiles.index:
        dst_path = basepath / fileset_path
        logger.info(f"Copying {img_path} to {dst_path}")
        shutil.copy(img_path, dst_path)


def distribute(input_dir, output_dir, reset):
    train_Name_path = input_dir / "train_infrared.txt"
    test_Name_path = input_dir / "test_infrared.txt"
    Image_path = input_dir / "JPEGImages"

    base_dir = Path(output_dir) / "soda"
    if reset and base_dir.exists():
        shutil.rmtree(base_dir)
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)

    main_dirs_image = ["image/train", "image/val", "image/test"]
    main_dirs_mask = ["mask/train", "mask/val", "mask/test"]

    for main in main_dirs_image + main_dirs_mask:
        path = base_dir / main
        if not path.exists():
            path.mkdir(parents=True)

    imageid_path_dict = {
        Path(x).stem: x for x in glob(str(Image_path / "**/*.jpg"), recursive=True)
    }

    def process_data(df, main_dirs_image, main_dirs_mask):
        df = df[~df.Image_Name.str.contains(r"\(")]  # Remove copies
        df = df[~df.Image_Name.str.contains(r"\~")]  # Remove copies
        for img_set, mask_set in zip(main_dirs_image, main_dirs_mask):
            copying(
                tiles=df,
                path_label="Image_Path",
                basepath=base_dir,
                fileset_path=img_set,
            )
            copying(
                tiles=df,
                path_label="Mask_Path",
                basepath=base_dir,
                fileset_path=mask_set,
            )

    # Process train set
    train_df = pd.read_table(train_Name_path, sep=r"\s+", names=("Image_Name", "B"))
    train_df["Image_Path"] = train_df["Image_Name"].map(imageid_path_dict.get)
    train_df["Mask_Path"] = train_df["Image_Path"].str.replace(".jpg", ".png")
    train_df["Mask_Path"] = train_df["Mask_Path"].str.replace("JPEGImages", "SegmentationClassOne")
    process_data(train_df, [main_dirs_image[0]], [main_dirs_mask[0]])

    # Process test and validation sets
    test_df = pd.read_table(test_Name_path, sep=r"\s+", names=("Image_Name", "B"))
    test_df["Image_Path"] = test_df["Image_Name"].map(imageid_path_dict.get)
    test_df["Mask_Path"] = test_df["Image_Path"].str.replace(".jpg", ".png")
    test_df["Mask_Path"] = test_df["Mask_Path"].str.replace("JPEGImages", "SegmentationClassOne")

    val_idx, test_idx = stratify(test_df["Mask_Path"])
    process_data(test_df.iloc[val_idx], [main_dirs_image[1]], [main_dirs_mask[1]])
    process_data(test_df.iloc[test_idx], [main_dirs_image[2]], [main_dirs_mask[2]])


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[RichHandler()],
    )

    parser = argparse.ArgumentParser(
        description="Distribute train, validation, and test sets for SODA dataset"
    )
    parser.add_argument(
        "--input-image-path",
        type=Path,
        default="./data/original/SODA/InfraredSemanticLabel/",
        help="Path to the SODA dataset images. This should lead contain the directory InfraredSemanticLabel.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default="data/processed_dataset/",
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Flag indicating whether to reset (remove existing) dataset directory if it already exists. True to reset, False to append to existing directory.",
    )

    args = parser.parse_args()
    distribute(
        input_dir=args.input_image_path,
        output_dir=args.save_path,
        reset=args.reset,
    )
