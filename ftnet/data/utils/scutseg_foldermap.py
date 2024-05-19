# =============================================================================
# To distribute train, valid, test for SODA
# =============================================================================

import argparse
import os
import shutil
from glob import glob

import pandas as pd


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
    # count=0
    for img_path in tiles.index:
        print(f"Path = {img_path}")
        dst_path = os.path.join(basepath, fileset_path)

        shutil.copy(img_path, dst_path)


# %%
# =============================================================================
# Creating folders
# =============================================================================


def distribute(input_dir, output_dir, reset):
    # basepath = './Dataset/SCUT/'
    basepath = input_dir

    train_Name_path = basepath + "scut_train.txt"
    val_Name_path = basepath + "scut_val.txt"
    Image_path = basepath + "images/"
    Mask_path = basepath + "gt_instance/"

    # os.path.abspath(os.path.join(basepath,'..'))
    base_dir = os.path.join(output_dir, "SCUTSEG")
    if reset and os.path.exists(base_dir):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    main_dirs_image = ["image/train", "image/val"]
    main_dirs_mask = ["mask/train", "mask/val"]

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
        for x in glob(os.path.join(Image_path, "**", "**", "*.jpg"), recursive=True)
    }
    maskid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(Mask_path, "**", "**", "*.jpg"), recursive=True)
    }

    train_df = pd.read_table(
        train_Name_path, delim_whitespace=True, names=("Image_subpath", "Mask_subpath")
    )
    train_df = train_df.sort_values(
        by="Image_subpath", axis=0, ascending=True, kind="quicksort"
    ).reset_index(drop=True)
    train_df = train_df.fillna("NA")

    train_df["Image_Name"] = train_df["Image_subpath"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    train_df["Image_Path"] = train_df["Image_Name"].map(imageid_path_dict.get)
    train_df["Mask_Path"] = train_df["Image_Name"].map(maskid_path_dict.get)

    train_df["Mask_Path"] = train_df["Mask_Path"].apply(
        lambda x: x.split(".")[0] + "_labelIds.png"
    )
    train_df1 = train_df.copy(deep=True)

    copying(
        tiles=train_df,
        path_label="Image_Path",
        basepath=base_dir,
        fileset_path=main_dirs_image[0],
    )
    copying(
        tiles=train_df1,
        path_label="Mask_Path",
        basepath=base_dir,
        fileset_path=main_dirs_mask[0],
    )

    validation_df = pd.read_table(
        val_Name_path, delim_whitespace=True, names=("Image_subpath", "Mask_subpath")
    )
    validation_df = validation_df.sort_values(
        by="Image_subpath", axis=0, ascending=True, kind="quicksort"
    ).reset_index(drop=True)
    validation_df = validation_df.fillna("NA")

    validation_df["Image_Name"] = validation_df["Image_subpath"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    validation_df["Image_Path"] = validation_df["Image_Name"].map(imageid_path_dict.get)
    validation_df["Mask_Path"] = validation_df["Image_Name"].map(maskid_path_dict.get)
    # validation_df['Mask_Path'] = validation_df['Mask_Path'].apply(lambda x: x+'_labelIds')
    validation_df["Mask_Path"] = validation_df["Mask_Path"].apply(
        lambda x: x.split(".")[0] + "_labelIds.png"
    )

    validation_df1 = validation_df.copy(deep=True)

    copying(
        tiles=validation_df,
        path_label="Image_Path",
        basepath=base_dir,
        fileset_path=main_dirs_image[1],
    )
    copying(
        tiles=validation_df1,
        path_label="Mask_Path",
        basepath=base_dir,
        fileset_path=main_dirs_mask[1],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up Cityscape Dataset")
    parser.add_argument(
        "--input-image-path",
        type=str,
        default="/mnt/1842213842211C4E/raw_dataset/SCUT-SEG/",
        help="Path to Scut-Seg Dataset",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/mnt/1842213842211C4E/processed_dataset/",
        help="Path to save Scut-Seg Dataset",
    )
    parser.add_argument(
        "--reset", type=bool, default=True, help="Path to Cityscape Dataset"
    )

    args = parser.parse_args()
    distribute(
        input_dir=args.input_image_path, output_dir=args.save_path, reset=args.reset
    )
