# =============================================================================
# To distribute train, valid, test for MFN
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


def copying(tiles, path_label, destpath, fileset_path):
    tiles.set_index(path_label, inplace=True)
    for img_path in tiles.index:
        print(f"Path = {img_path}")
        dst_path = os.path.join(destpath, fileset_path)

        shutil.copy(img_path, dst_path)


def create_df(txt_path, imageid_path_dict, blacklist_df):
    df = pd.read_table(txt_path, delim_whitespace=True, names=("Image_Name", "B"))
    df = df.sort_values(
        by="Image_Name", axis=0, ascending=True, kind="quicksort"
    ).reset_index(drop=True)
    df = df.fillna("NA")
    del df["B"]
    df["Image_Path"] = df["Image_Name"].map(imageid_path_dict.get)
    df["Mask_Path"] = df["Image_Path"].str.replace("images", "labels")

    df = df[
        ~df.Image_Name.str.contains(r"\(")
    ]  # there are copies of some files, eg, ABCD.png and ABCD (1).png (I am removing the copies)
    df = df[
        ~df.Image_Name.str.contains(r"\~")
    ]  # there are copies of some files, eg, ABCD.png and ~temp_ABCD.png (I am removing the copies)
    df = df[
        ~df.Image_Name.str.contains("flip")
    ]  # there are copies of some files, eg, ABCD.png and ABCD (1).png (I am removing the copies)
    cond = df["Image_Name"].isin(blacklist_df["Image_Name"])
    df.drop(df[cond].index, inplace=True)

    return df


def copy_files_from_df(df, destpath, main_dirs_image, main_dirs_mask):
    copying(
        tiles=df,
        path_label="Image_Path",
        destpath=destpath,
        fileset_path=main_dirs_image,
    )
    copying(
        tiles=df, path_label="Mask_Path", destpath=destpath, fileset_path=main_dirs_mask
    )


def distribute(input_dir, output_dir, reset):
    basepath = input_dir
    # basepath = "./Thermal_Segmentation/Dataset/ir_seg_dataset-20210510T225620Z-001"
    # destpath = os.path.join(basepath, DATASET)
    # reset = True

    train_txt_path = basepath + "/train.txt"
    validation_txt_path = basepath + "/val.txt"
    test_txt_path = basepath + "/test.txt"
    blacklist_path = basepath + "/black_list.txt"

    Image_path = basepath + "/images"
    # Mask_path = basepath + "/labels"
    destpath = os.path.join(output_dir, "MFN")

    # os.path.abspath(os.path.join(basepath,'..'))
    if reset and os.path.exists(destpath):
        if os.path.exists(destpath):
            shutil.rmtree(destpath)
    if not os.path.exists(destpath):
        os.mkdir(destpath)

    main_dirs_image = ["image/train", "image/val", "image/test"]
    main_dirs_mask = ["mask/train", "mask/val", "mask/test"]

    for main in main_dirs_image:
        path = os.path.join(destpath, main)
        if not os.path.exists(path):
            os.makedirs(path)

    for main in main_dirs_mask:
        path = os.path.join(destpath, main)
        if not os.path.exists(path):
            os.makedirs(path)

    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(Image_path, "**", "**", "*.png"), recursive=True)
    }

    blacklist_df = pd.read_table(
        blacklist_path, delim_whitespace=True, names=("Image_Name", "B")
    )

    train_df = create_df(train_txt_path, imageid_path_dict, blacklist_df)
    validation_df = create_df(validation_txt_path, imageid_path_dict, blacklist_df)
    test_df = create_df(test_txt_path, imageid_path_dict, blacklist_df)

    copy_files_from_df(train_df, destpath, main_dirs_image[0], main_dirs_mask[0])
    copy_files_from_df(validation_df, destpath, main_dirs_image[1], main_dirs_mask[1])
    copy_files_from_df(test_df, destpath, main_dirs_image[2], main_dirs_mask[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up Cityscape Dataset")
    parser.add_argument(
        "--input-image-path",
        type=str,
        default="/mnt/1842213842211C4E/raw_dataset/ir_seg_dataset/",
        help="Path to MFN Dataset",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/mnt/1842213842211C4E/processed_dataset/",
        help="Path to save MFN Dataset",
    )
    parser.add_argument(
        "--reset", type=bool, default=True, help="Path to Cityscape Dataset"
    )

    args = parser.parse_args()
    distribute(
        input_dir=args.input_image_path, output_dir=args.save_path, reset=args.reset
    )
