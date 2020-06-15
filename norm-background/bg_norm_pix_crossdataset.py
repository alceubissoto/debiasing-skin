import os
import glob
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

import cv2
from PIL import Image
import argparse

parser = argparse.ArgumentParser('argument for normalizing the background of the dataset')

# We use the normalized background obtained on train to apply on train and test.
parser.add_argument('--train_image_path', type=str, help='Path of the training 299x299x3 images')
parser.add_argument('--train_mask_path', type=str, help='Path of the training 299x299 masks')
parser.add_argument('--train_csv_path', type=str, help='csv which contain the images used to normalize')

parser.add_argument('--test_image_path', type=str, help='Path of the test 299x299x3 images')
parser.add_argument('--test_mask_path', type=str, help='Path of the test 299x299 masks')
parser.add_argument('--test_csv_path', type=str, help='csv which contain the images used to normalize')

parser.add_argument('--output_path', type=str, help='Path to save the normalized images')

args = parser.parse_args()

def verify_binary_mask(image):
    """ Returns an verified image with pixel values != 0(before) truncated to 1(after)
        
        Returns:
        image (np.array): Array with binary pixel values, either 0 (skin) or 1 (lesion)
    """
    return np.ceil(image)

def normalize_image_background(img_list, img_data_path, gt_data_path):
    """ Calculate the sum of pixel values for the background and the number of images that contributes for that pixel
        
        Returns: 
        bg_sum (np.array): Array with the pixels summed
        bg_count (np.array): Array with number of the pixels that contributed to the sum.
    
    """
    
    ### The images need to already be in 299x299 shape, both traditional and segmentation
    res = (299, 299)
    #bg_sum = np.zeros((3))
    bg_sum = np.zeros((res[0], res[1], 3), dtype=np.float)
    bg_count = np.zeros((res[0], res[1], 3), dtype=np.int)

    num_img = len(img_list)

    for n in tqdm(range(0, num_img)):
        img_name = img_list[n]
        
        img = Image.open(img_data_path + img_name + ".png")
        img = np.array(img)

        gt = Image.open(gt_data_path + img_name + ".png")
        gt = np.array(gt)
        gt = verify_binary_mask(gt) ### Ceils the pixels that are between 0 and 1

        for i in range(res[0]):
            for j in range(res[1]):
                if gt[i][j] == 0:
                    bg_sum[i][j] += img[i][j]
                    bg_count[i][j] += 1

    return bg_sum, bg_count

#----------------


traditional_path = args.train_image_path # original skin lesion images resized to 299x299 
segmentation_path = args.train_mask_path # original binary mask images resized to 299x299

all_dataset_csv = args.train_csv_path

### Normalize All ISIC Train Dataset 
 
print("Normalizing All ISIC Dataset: {}".format(all_dataset_csv))

isic_df = pd.read_csv(all_dataset_csv, delimiter = ';', index_col=None)

output_path = args.output_path
print("Output Path: {}".format(output_path))

if not os.path.exists(output_path):
    os.mkdir(output_path)

res = 299, 299
img_list = []


# Add all split images and masks to a list
for idx in range(0, isic_df.shape[0]):
    img_name = isic_df.iloc[idx][0]
    img_list.append(img_name)


### Calculates the background normalization using train
bg_sum, bg_count = normalize_image_background(img_list, traditional_path, segmentation_path)
print("Background Normalization: Done!")
norm = bg_sum / bg_count
# 'norm' contain the normalized background
norm=np.array(np.round(norm),dtype=np.uint8)


# Apply normalization to train
for idx in tqdm(range(len(img_list))):
    img_name = img_list[idx]

    img = Image.open(traditional_path + img_name + ".png")
    img = np.array(img)

    gt = Image.open(segmentation_path + img_name + ".png")
    gt = np.array(gt)
    gt = np.dstack([gt]*3)

    combination = np.where(gt > 0, img, norm)
    
    print("Normalized Image: {}".format(output_path + img_name))
    combination = Image.fromarray(combination)
    combination.save(output_path + img_name + ".png")

### Applying the train set pixel average in images of the test set
print("\nApplying The Normalization in Atlas Test Set")

atlas_traditional_path = args.test_image_path
atlas_segmentation_path = args.test_mask_path
atlas_csv_path = args.test_csv_path

atlas_output_path = args.output_path

atlas_df = pd.read_csv(atlas_csv_path, delimiter=";", index_col=None)

img_list = []
gt_list = []

for idx in range(0, atlas_df.shape[0]):
    img_name = atlas_df.iloc[idx][2]
    img_list.append(img_name)

    gt_name = img_name
    gt_list.append(gt_name)

for idx in range(len(img_list)):
    img_name = img_list[idx]
    gt_name = gt_list[idx]

    img = Image.open(atlas_traditional_path + img_name + ".png")
    img = np.array(img)

    gt = Image.open(atlas_segmentation_path + gt_name + ".png")
    gt = np.array(gt)
    gt = np.dstack([gt]*3)
    
    combination = np.where(gt > 0, img, norm)


    print("Normalized Atlas Image: {}".format(output_path + img_name))
    combination = Image.fromarray(combination)
    combination.save(output_path + img_name + ".png")

