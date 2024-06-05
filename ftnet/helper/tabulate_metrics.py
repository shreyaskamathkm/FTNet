# =============================================================================
# To distribute train, valid, test for SODA
# =============================================================================

import argparse
import os
import shutil


def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in [".txt"])


#
# =============================================================================
# Copying files to different folders
# =============================================================================
def copying(tiles, path_label, basepath, fileset_path):
    tiles.set_index(path_label, inplace=True)
    for img_path in tiles.index:
        print(f"Path = {img_path}")
        dst_path = os.path.join(basepath, fileset_path)

        shutil.copy(img_path, dst_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="the path to list")
    args = parser.parse_args()
    # tabulate_metrics(args.directory)
