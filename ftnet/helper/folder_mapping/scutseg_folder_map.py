"""
Example run:   python -m ftnet.helper.folder_mapping.scutseg_folder_map   --input-image-path <path to original dataset>  --save-path <path to save dataset>
python -m ftnet.helper.folder_mapping.scutseg_folder_map  --input-image-path ./data/original/scutseg/ --save-path ./data/processed_dataset/
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Union

import pandas as pd
from rich.logging import RichHandler

# Set up logging
logger = logging.getLogger(__name__)


def copying(tiles: pd.DataFrame, path_label: str, basepath: Path, fileset_path: str) -> None:
    """Copy files from their source paths to the destination directory."""
    tiles.set_index(path_label, inplace=True)
    dst_path = basepath / fileset_path
    dst_path.mkdir(parents=True, exist_ok=True)

    for img_path in tiles.index:
        logger.info(f"Copying {img_path} to {dst_path}")
        new_filename = Path(img_path).name.replace("_labelIds.png", ".png")

        shutil.copy2(img_path, dst_path / new_filename)


def process_split(
    split: str,
    name_path: Path,
    main_dir_image: str,
    main_dir_mask: str,
    imageid_path_dict: dict,
    maskid_path_dict: dict,
    base_dir: Path,
) -> None:
    """Process a dataset split (train/val) and copy images and masks to their
    respective directories."""
    df = pd.read_table(name_path, delim_whitespace=True, names=("Image_subpath", "Mask_subpath"))
    df = df.sort_values(by="Image_subpath").reset_index(drop=True)
    df["Image_Name"] = df["Image_subpath"].apply(lambda x: Path(x).stem)
    df["Image_Path"] = df["Image_Name"].map(imageid_path_dict.get)
    df["Mask_Path"] = (
        df["Image_Name"]
        .map(maskid_path_dict.get)
        .apply(lambda x: str(x).replace(".jpg", "_labelIds.png"))
    )

    copying(df, "Image_Path", base_dir, main_dir_image)
    copying(df, "Mask_Path", base_dir, main_dir_mask)


def distribute(input_dir: Union[str, Path], output_dir: Union[str, Path], reset: bool) -> None:
    """Distribute images and masks from input directory to the output
    directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    base_dir = output_dir / "scutseg"
    if reset and base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    main_dirs_image = ["image/train", "image/val"]
    main_dirs_mask = ["mask/train", "mask/val"]

    for main in main_dirs_image + main_dirs_mask:
        (base_dir / main).mkdir(parents=True, exist_ok=True)

    imageid_path_dict = {p.stem: p for p in input_dir.glob("images/**/*.jpg")}
    maskid_path_dict = {p.stem: p for p in input_dir.glob("gt_instance/**/*.jpg")}

    process_split(
        "train",
        input_dir / "scut_train.txt",
        main_dirs_image[0],
        main_dirs_mask[0],
        imageid_path_dict,
        maskid_path_dict,
        base_dir,
    )
    process_split(
        "val",
        input_dir / "scut_val.txt",
        main_dirs_image[1],
        main_dirs_mask[1],
        imageid_path_dict,
        maskid_path_dict,
        base_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[RichHandler()],
    )

    parser = argparse.ArgumentParser(
        description="Distribute train, validation, and test sets for SCUTSEG Dataset"
    )
    parser.add_argument(
        "--input-image-path",
        type=Path,
        default="./data/original/scutseg/",
        help="Path to the mfn dataset images. This path should contain the path to scutseg",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default="./data/processed_dataset",
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Flag indicating whether to reset (remove existing) dataset directory if it already exists.",
    )

    args = parser.parse_args()
    distribute(input_dir=args.input_image_path, output_dir=args.save_path, reset=args.reset)
