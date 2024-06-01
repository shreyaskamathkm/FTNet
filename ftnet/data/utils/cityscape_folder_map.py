import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from rich.logging import RichHandler

# Set up logging
logger = logging.getLogger(__name__)


def copying(tiles: pd.DataFrame, path_label: str, basepath: Path, fileset_path: str) -> None:
    """Copy files from their source paths to the destination directory.

    Args:
    tiles (pd.DataFrame): DataFrame containing file paths.
    path_label (str): Column name containing the file paths.
    basepath (Path): Base directory for the destination.
    fileset_path (str): Subdirectory for the destination.
    """
    tiles.set_index(path_label, inplace=True)
    dst_path = basepath / fileset_path
    dst_path.mkdir(parents=True, exist_ok=True)

    for img_path in tiles.index:
        logger.info(f"Copying {img_path} to {dst_path}")
        shutil.copy(img_path, dst_path)


def process_split(split: str, input_dir: Path, save_dir: Path, reset: bool) -> None:
    """Process and copy files for a specific dataset split (train, val, test).

    Args:
    split (str): The dataset split to process (train, val, test).
    input_dir (Path): Path to the input images directory.
    save_dir (Path): Path to the output base directory.
    reset (bool): Flag to reset the output directory.
    """
    split_dirs = [f"cityscape/image/{split}", f"cityscape/mask/{split}"]

    if reset and save_dir.exists():
        shutil.rmtree(save_dir)

    for main in split_dirs:
        path = save_dir / main
        path.mkdir(parents=True, exist_ok=True)

    imageid_path_dict = {Path(x).stem: x for x in Path(input_dir, split).rglob("*.jpg")}

    tile_df = pd.DataFrame(imageid_path_dict.items(), columns=["Image_Name", "Image_Path"])
    tile_df = tile_df.sort_values(by="Image_Name").reset_index(drop=True)
    tile_df["Mask_Path"] = tile_df["Image_Path"].apply(lambda x: str(x).replace(".jpg", ".png"))
    tile_df["Mask_Path"] = tile_df["Mask_Path"].apply(
        lambda x: str(x).replace("TIR_leftImg8bit", "TIR_leftImg8bit/gtFine")
    )
    tile_df["Mask_Path"] = tile_df["Mask_Path"].apply(
        lambda x: str(x).replace("leftImg8bit_synthesized_image", "gtFine_labelIds")
    )
    tile_df["Mask_Name"] = tile_df["Image_Name"].apply(
        lambda x: str(x).replace("leftImg8bit_synthesized_image", "gtFine_labelIds")
    )
    tile_df = tile_df[~tile_df.Image_Name.str.contains(r"[\(\)~]")]

    copying(tile_df, "Image_Path", save_dir, split_dirs[0])
    copying(tile_df, "Mask_Path", save_dir, split_dirs[1])


def distribute(input_dir: Path, output_dir: Path, reset: bool) -> None:
    """Distribute images and masks from input directory to the output
    directory.

    Args:
    input_dir (Path): Path to the input images directory.
    output_dir (Path): Path to the output base directory.
    reset (bool): Flag to reset the output directory.
    """
    splits = ["train", "val", "test"]
    for split in splits:
        # split_save_dir = output_dir / f"cityscape/ {split}"
        process_split(split, input_dir, output_dir, reset)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[RichHandler()],
    )

    parser = argparse.ArgumentParser(description="Set up Cityscape Dataset")
    parser.add_argument(
        "--input-image-path",
        type=Path,
        default=Path("/mnt/C26EDFBB6EDFA687/lab-work/FTNet/datasets/SODA/SODA/TIR_leftImg8bit/"),
        help="Path to the Cityscape dataset images. This path should lead to the directory containing TIR_leftImg8bit images.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("/mnt/C26EDFBB6EDFA687/lab-work/FTNet/data/processed_dataset/"),
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Flag indicating whether to reset (remove existing) dataset directory if it already exists.",
    )

    args = parser.parse_args()
    distribute(input_dir=args.input_image_path, output_dir=args.save_path, reset=args.reset)
