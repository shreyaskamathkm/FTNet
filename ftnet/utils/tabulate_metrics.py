# =============================================================================
# To distribute train, valid, test for SODA
# =============================================================================

import os
import shutil
import argparse


def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in [".txt"])


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
        print(f"Path = {img_path}")
        dst_path = os.path.join(basepath, fileset_path)

        shutil.copy(img_path, dst_path)


# %%
# =============================================================================
# Creating folders
# =============================================================================


# def tabulate_metrics(directory):
#     subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
#     for x in glob(
#         os.path.join(subfolders, "soda/Segmented images/test/", "Average.txt")
#     ):
#         with open(x) as f:
#             lines = f.readlines()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="the path to list")
    args = parser.parse_args()
    # tabulate_metrics(args.directory)
