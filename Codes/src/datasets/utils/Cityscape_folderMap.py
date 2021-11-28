# =============================================================================
# To distribute train, valid, test for SODA
# =============================================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
import shutil
import argparse


def is_image_file(filename):
    return any(
        filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"]
    )


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# %%
# =============================================================================
# Copying files to different folders
# =============================================================================
def copying(tiles, path_label, basepath, fileset_path):
    tiles.set_index(path_label, inplace=True)
    for img_path in tiles.index:
        print("Path = {}".format(img_path))
        dst_path = os.path.join(basepath, fileset_path)

        shutil.copy(img_path, dst_path)


# %%
# =============================================================================
# Creating folders
# =============================================================================


def distribute(input_dir, output_dir, reset):
    # basepath = './Thermal_Segmentation/Dataset/Cityscapes_thermal/TIR_leftImg8bit/'
    basepath = input_dir

    # os.path.abspath(os.path.join(basepath,'..'))
    base_dir = os.path.join(output_dir, "CITYSCAPE_5000")
    if reset == True and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    main_dirs_image = ["image/train"]
    main_dirs_mask = ["mask/train"]

    for main in main_dirs_image:

        path = os.path.join(base_dir, main)
        if not os.path.exists(path):
            os.makedirs(path)

    for main in main_dirs_mask:

        path = os.path.join(base_dir, main)
        if not os.path.exists(path):
            os.makedirs(path)

    # %%
    # =============================================================================
    # Creating folders
    # =============================================================================

    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(basepath, "**", "**", "*.jpg"), recursive=True)
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

    # https://stackoverflow.com/questions/28679930/how-to-drop-rows-from-pandas-data-frame-that-contains-a-particular-string-in-a-p
    # https://stackoverflow.com/questions/41425945/python-pandas-error-missing-unterminated-subpattern-at-position-2
    tile_df = tile_df[
        ~tile_df.Image_Name.str.contains("\(")
    ]  # there are copies of some files, eg, ABCD.png and ABCD (1).png (I am removing the copies)
    # there are copies of some files, eg, ABCD.png and ~temp_ABCD.png (I am removing the copies)
    tile_df = tile_df[~tile_df.Image_Name.str.contains("\~")]

    copying(
        tiles=tile_df,
        path_label="Image_Path",
        basepath=base_dir,
        fileset_path=main_dirs_image[0],
    )
    copying(
        tiles=tile_df,
        path_label="Mask_Path",
        basepath=base_dir,
        fileset_path=main_dirs_mask[0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up Cityscape Dataset")
    parser.add_argument(
        "--input-image-path",
        type=str,
        default="/mnt/1842213842211C4E/raw_dataset/SODA-20211127T202136Z-001/SODA/TIR_leftImg8bit/",
        help="Path to Cityscape Dataset. Note: This should lead to TIR_leftImg8bit",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/mnt/1842213842211C4E/processed_dataset/",
        help="Path to Cityscape Dataset",
    )
    parser.add_argument(
        "--reset", type=bool, default=True, help="Path to Cityscape Dataset"
    )

    args = parser.parse_args()
    distribute(
        input_dir=args.input_image_path,
        output_dir=args.save_path,
        reset=args.reset,
    )
