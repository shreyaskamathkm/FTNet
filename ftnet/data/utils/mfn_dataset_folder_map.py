import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rich.logging import RichHandler

# Set up logging
logger = logging.getLogger(__name__)


def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def copying(tiles: pd.DataFrame, path_label: str, destpath: Path, fileset_path: str) -> None:
    tiles.set_index(path_label, inplace=True)
    for img_path in tiles.index:
        dst_path = destpath / fileset_path
        logger.info(f"Copying {img_path} to {dst_path}")
        shutil.copy(img_path, dst_path)


def create_df(
    txt_path: Path, imageid_path_dict: Dict[str, Path], blacklist_df: pd.DataFrame
) -> pd.DataFrame:
    df = pd.read_table(txt_path, delim_whitespace=True, names=("Image_Name", "B"))
    df.sort_values(by="Image_Name", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.fillna("NA", inplace=True)
    df.drop(columns=["B"], inplace=True)

    df["Image_Path"] = df["Image_Name"].map(imageid_path_dict.get)
    df["Mask_Path"] = df["Image_Path"].apply(lambda x: str(x).replace("images", "labels"))

    df = df[
        ~df["Image_Name"].str.contains(r"\(")
    ]  # there are copies of some files, eg, ABCD.png and ABCD (1).png (removing the copies)
    df = df[
        ~df["Image_Name"].str.contains(r"\~")
    ]  # there are copies of some files, eg, ABCD.png and ~temp_ABCD.png (removing the copies)
    df = df[
        ~df["Image_Name"].str.contains("flip")
    ]  # there are copies of some files, eg, ABCD.png and ABCD (1).png (removing the copies)

    cond = df["Image_Name"].isin(blacklist_df["Image_Name"])
    df.drop(df[cond].index, inplace=True)

    return df


def copy_files_from_df(
    df: pd.DataFrame, destpath: Path, main_dirs_image: str, main_dirs_mask: str
) -> None:
    copying(df, "Image_Path", destpath, main_dirs_image)
    copying(df, "Mask_Path", destpath, main_dirs_mask)


def setup_directories(destpath: Path, subdirs: List[str]) -> None:
    for subdir in subdirs:
        path = destpath / subdir
        path.mkdir(parents=True, exist_ok=True)


def distribute(input_dir: str, output_dir: str, reset: bool) -> None:
    basepath = Path(input_dir)
    destpath = Path(output_dir) / "mfn"

    if reset and destpath.exists():
        shutil.rmtree(destpath)
    destpath.mkdir(parents=True, exist_ok=True)

    main_dirs_image = ["image/train", "image/val", "image/test"]
    main_dirs_mask = ["mask/train", "mask/val", "mask/test"]

    setup_directories(destpath, main_dirs_image + main_dirs_mask)

    image_paths = basepath.glob("images/**/*.png")
    imageid_path_dict = {path.stem: path for path in image_paths}

    blacklist_df = pd.read_table(
        basepath / "black_list.txt", delim_whitespace=True, names=("Image_Name", "B")
    )

    for split, subdir_image, subdir_mask in zip(
        ["train", "val", "test"], main_dirs_image, main_dirs_mask
    ):
        txt_path = basepath / f"{split}.txt"
        df = create_df(txt_path, imageid_path_dict, blacklist_df)
        copy_files_from_df(df, destpath, subdir_image, subdir_mask)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[RichHandler()],
    )

    parser = argparse.ArgumentParser(description="Set up MFN Dataset")
    parser.add_argument(
        "--input-image-path",
        type=str,
        default="/home/shreyas/Downloads/ir_seg_dataset-20240531T202528Z-001/ir_seg_dataset",
        help="Path to MFN Dataset",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/mnt/C26EDFBB6EDFA687/lab-work/FTNet/data/processed_dataset/",
        help="Path to save MFN Dataset",
    )
    parser.add_argument("--reset", type=bool, default=True, help="Reset the dataset")

    args = parser.parse_args()
    distribute(input_dir=args.input_image_path, output_dir=args.save_path, reset=args.reset)
