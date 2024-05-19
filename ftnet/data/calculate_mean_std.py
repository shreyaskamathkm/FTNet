import argparse
import glob
import os
import time
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
import tqdm


def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return (
        im.astype(float) / info.max
    )  # Divide all values by the largest possible value in the datatype


parser = argparse.ArgumentParser(description="Finding Mean and Std Deviation")
parser.add_argument(
    "--test_folder",
    type=str,
    default="./Dataset/SCUT/SCUTSEG/image/train/",
    help="Dataset for checking the model",
)
args = parser.parse_args()
args.test_folder = args.test_folder.replace("\\", "/")


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif", "tiff"]
    )


image_list = [
    x
    for x in glob.glob(os.path.join(args.test_folder, "*"), recursive=True)
    if is_image_file(x)
]


def caluate_MSTD(paths):
    img = cv2.imread(paths, 1)
    img = im2double(img)
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    return mean, std


start = time.time()
with ThreadPool(6) as p:
    patch_grids = list(
        tqdm.tqdm(p.imap(caluate_MSTD, image_list), total=len(image_list))
    )

all_img_means = [x[0] for x in patch_grids]
all_img_std = [x[1] for x in patch_grids]

mean_per_channel = np.mean(np.array(all_img_means), axis=0)
std_per_channel = np.mean(np.array(all_img_std), axis=0)


print(f"Total Time = {time.time() - start}")

print(mean_per_channel)
print(std_per_channel)

np.savez("./mfn", mean=mean_per_channel, std=std_per_channel)
