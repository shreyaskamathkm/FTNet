import shutil
import argparse
import pandas as pd
from pathlib import Path
from glob import glob


def copying(tiles, path_label, basepath, fileset_path):
    tiles.set_index(path_label, inplace=True)
    for img_path in tiles.index:
        print(f"Path = {img_path}")
        dst_path = basepath / fileset_path
        shutil.copy(img_path, dst_path)


def distribute(input_dir, output_dir, reset):
    basepath = Path(input_dir)
    base_dir = Path(output_dir) / "CITYSCAPE_5000"

    if reset and base_dir.exists():
        shutil.rmtree(base_dir)

    if not base_dir.exists():
        base_dir.mkdir(parents=True)

    main_dirs = ["image/train", "mask/train"]

    for main in main_dirs:
        path = base_dir / main
        if not path.exists():
            path.mkdir(parents=True)

    imageid_path_dict = {
        Path(x).stem: x for x in glob(str(basepath / "**/*.jpg"), recursive=True)
    }

    tile_df = pd.DataFrame(
        imageid_path_dict.items(), columns=["Image_Name", "Image_Path"]
    )
    tile_df = tile_df.sort_values(
        by="Image_Name", axis=0, ascending=True, kind="quicksort"
    ).reset_index(drop=True)
    tile_df = tile_df.fillna("NA")
    tile_df["Mask_Path"] = tile_df["Image_Path"].str.replace(".jpg", ".png")
    tile_df["Mask_Path"] = tile_df["Mask_Path"].str.replace(
        "TIR_leftImg8bit", "TIR_leftImg8bit/gtFine"
    )
    tile_df["Mask_Path"] = tile_df["Mask_Path"].str.replace(
        "leftImg8bit_synthesized_image", "gtFine_labelIds"
    )
    tile_df["Mask_Name"] = tile_df["Image_Name"].str.replace(
        "leftImg8bit_synthesized_image", "gtFine_labelIds"
    )

    tile_df = tile_df[~tile_df.Image_Name.str.contains(r"\(")]
    tile_df = tile_df[~tile_df.Image_Name.str.contains(r"\~")]

    copying(
        tiles=tile_df,
        path_label="Image_Path",
        basepath=base_dir,
        fileset_path=main_dirs[0],
    )
    copying(
        tiles=tile_df,
        path_label="Mask_Path",
        basepath=base_dir,
        fileset_path=main_dirs[1],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up Cityscape Dataset")
    parser.add_argument(
        "--input-image-path",
        type=str,
        default="/mnt/1842213842211C4E/raw_dataset/SODA-20211127T202136Z-001/SODA/TIR_leftImg8bit/",
        help="Path to the Cityscape dataset images. This path should lead to the directory containing TIR_leftImg8bit images.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/mnt/1842213842211C4E/processed_dataset/",
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
