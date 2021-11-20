import os
import glob
from core.utils.visualize import get_color_pallete
import cv2
import matplotlib.pyplot as plt
import numpy as np

mask_path = '/mnt/ACF29FC2F29F8F68/Work/Deep_Learning/Thermal_Segmentation/Dataset/InfraredSemanticLabel-20210430T150555Z-001/SODA/mask/test/'
mask_color_path = '/mnt/ACF29FC2F29F8F68/Work/Deep_Learning/Thermal_Segmentation/Dataset/InfraredSemanticLabel-20210430T150555Z-001/SODA/mask_color/test/'


if not os.path.exists(mask_color_path):
    os.makedirs(mask_color_path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif", "tiff"])


masks = {os.path.splitext(os.path.basename(x))[0]: x for x in glob.glob(
    os.path.join(mask_path, '*'), recursive=True) if is_image_file(x)}


for name, path in masks.items():
    img = cv2.imread(path)[:, :, 0]
    temp = get_color_pallete(img, dataset='soda')
    plt.imsave(mask_color_path +
               '/original_mask_{}.png'.format(name), np.array(temp))
