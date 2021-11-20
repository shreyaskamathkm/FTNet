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
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# %%
# =============================================================================
# Copying files to different folders
# =============================================================================
def copying(tiles, path_label, basepath, fileset_path):
    tiles.set_index(path_label, inplace=True)
    for img_path in tiles.index:
        print('Path = {}'.format(img_path))
        dst_path = os.path.join(basepath, fileset_path)

        shutil.copy(img_path, dst_path)

# %%
# =============================================================================
# Creating folders
# =============================================================================


basepath = './Thermal_Segmentation/Dataset/InfraredSemanticLabel-20210430T150555Z-001/'
reset = True

train_Name_path = basepath + '/InfraredSemanticLabel/train_infrared.txt'
test_Name_path = basepath + '/InfraredSemanticLabel/test_infrared.txt'
Image_path = basepath + '/InfraredSemanticLabel/JPEGImages'
Mask_path = basepath + '/InfraredSemanticLabel/SegmentationClassOne'

# os.path.abspath(os.path.join(basepath,'..'))
base_dir = os.path.join(basepath, 'SODA')
if reset == True:
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

main_dirs_image = ['image/train', 'image/val', 'image/test']
main_dirs_mask = ['mask/train', 'mask/val', 'mask/test']

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


imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(Image_path, '**', '**', '*.jpg'), recursive=True)}


name_df = pd.read_table(test_Name_path, delim_whitespace=True, names=('Image_Name', 'B'))
name_df = name_df.sort_values(by='Image_Name', axis=0, ascending=True, kind='quicksort').reset_index(drop=True)
name_df = name_df.fillna('NA')
del name_df['B']
name_df['Image_Path'] = name_df['Image_Name'].map(imageid_path_dict.get)
name_df['Mask_Path'] = name_df['Image_Path'].str.replace('.jpg', '.png')
name_df['Mask_Path'] = name_df['Mask_Path'].str.replace('JPEGImages', 'SegmentationClassOne')


test_df, validation_df = train_test_split(name_df, test_size=0.1, random_state=2)

test_df = test_df[~test_df.Image_Name.str.contains("\(")]  # there are copies of some files, eg, ABCD.png and ABCD (1).png (I am removing the copies)
# there are copies of some files, eg, ABCD.png and ~temp_ABCD.png (I am removing the copies)
test_df = test_df[~test_df.Image_Name.str.contains("\~")]
# there are copies of some files, eg, ABCD.png and ABCD (1).png (I am removing the copies)
validation_df = validation_df[~validation_df.Image_Name.str.contains("\(")]
# there are copies of some files, eg, ABCD.png and ~temp_ABCD.png (I am removing the copies)
validation_df = validation_df[~validation_df.Image_Name.str.contains("\~")]


validation_df1 = validation_df.copy(deep=True)


copying(tiles=validation_df, path_label='Image_Path', basepath=base_dir, fileset_path=main_dirs_image[1])
copying(tiles=validation_df, path_label='Mask_Path', basepath=base_dir, fileset_path=main_dirs_mask[1])

copying(tiles=test_df, path_label='Image_Path', basepath=base_dir, fileset_path=main_dirs_image[2])
copying(tiles=test_df, path_label='Mask_Path', basepath=base_dir, fileset_path=main_dirs_mask[2])

# Include validation set in test as well
copying(tiles=validation_df1, path_label='Image_Path', basepath=base_dir, fileset_path=main_dirs_image[2])
copying(tiles=validation_df1, path_label='Mask_Path', basepath=base_dir, fileset_path=main_dirs_mask[2])
#


train_df = pd.read_table(train_Name_path, delim_whitespace=True, names=('Image_Name', 'B'))
train_df = train_df.sort_values(by='Image_Name', axis=0, ascending=True, kind='quicksort').reset_index(drop=True)
train_df = train_df.fillna('NA')
del train_df['B']
train_df['Image_Path'] = train_df['Image_Name'].map(imageid_path_dict.get)
train_df['Mask_Path'] = train_df['Image_Path'].str.replace('.jpg', '.png')
train_df['Mask_Path'] = train_df['Mask_Path'].str.replace('JPEGImages', 'SegmentationClassOne')
# https://stackoverflow.com/questions/28679930/how-to-drop-rows-from-pandas-data-frame-that-contains-a-particular-string-in-a-p
# https://stackoverflow.com/questions/41425945/python-pandas-error-missing-unterminated-subpattern-at-position-2
# there are copies of some files, eg, ABCD.png and ABCD (1).png (I am removing the copies)
train_df = train_df[~train_df.Image_Name.str.contains("\(")]
# there are copies of some files, eg, ABCD.png and ~temp_ABCD.png (I am removing the copies)
train_df = train_df[~train_df.Image_Name.str.contains("\~")]


copying(tiles=train_df, path_label='Image_Path', basepath=base_dir, fileset_path=main_dirs_image[0])
copying(tiles=train_df, path_label='Mask_Path', basepath=base_dir, fileset_path=main_dirs_mask[0])
