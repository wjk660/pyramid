# 1. Load the dataset
# 2. Use the albumentations library to augment the dataset.
from PIL import Image
import os
import cv2
from tqdm import tqdm
from glob import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from numba import njit
import matplotlib.pyplot as plt
# from albumentations import *
# ShiftScaleRotate,CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip,ElasticTransform

def load_data(path):
    images = sorted(glob.glob(f"{path}/*_img*"))
    masks = sorted(glob.glob(f"{path}/*_seg*"))
    return images, masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_data(images_path_list, masks_path_list, save_path, augment=True):
    for image_path, mask_path in tqdm(zip(images_path_list, masks_path_list), total=len(images)):
        """ Extracting the name and extension of the image and the mask. """
        image_name,image_extn = os.path.split(image_path)[1].split('.')  # os.path.split(x)可分隔路径和文件名
        mask_name,mask_extn = os.path.split(mask_path)[1].split('.')

        """ Reading image and mask. """
        image=Image.open(image_path)
        mask=Image.open(mask_path)
        # plt.imshow(image)
        # plt.show()
        H=image.size[0]
        W=image.size[1]
        """ Augmentation """
        save_images = [image]
        save_masks = [mask]
        if augment == True:
            for i in range(6):
                def aug(image, mask):
                    # 旋转
                    # angle = random.randint(-45, 45)
                    angle=(i+1)*15
                    image = TF.rotate(image, angle)
                    mask = TF.rotate(mask, angle)
                    # if i//2==0:
                    #     image = TF.hflip(image)
                    #     mask = TF.hflip(mask)
                    # else:
                    #     image = TF.vflip(image)
                    #     mask = TF.vflip(mask)
                    # more transforms
                    flip_rnd = random.random()
                    if flip_rnd > (1 - 0.3): #20%
                        image=TF.hflip(image)
                        mask = TF.hflip(mask)
                    if flip_rnd > (1 - 0.6): #20%
                        image = TF.vflip(image)
                        mask = TF.vflip(mask)
                    return image, mask
                x,y=aug(image,mask)
                # plt.imshow(x)
                # plt.show()
                save_images.append(x)
                save_masks.append(y)
        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"
            else:
                tmp_img_name = f"{image_name[:-4]}_{idx}_img.{image_extn}"
                tmp_mask_name = f"{mask_name[:-4]}_{idx}_seg.{mask_extn}"

            image_path = os.path.join(save_path, tmp_img_name)
            mask_path = os.path.join(save_path, tmp_mask_name)
            i.save(image_path)
            m.save(mask_path)
            idx += 1

if __name__ == "__main__":
    """ Loading original images and masks. """
    path = "/home/wangjk/project/pyramid/data/public_originPic"
    images, masks = load_data(path)
    print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

    """ Creating folders. """
    new_data_path="/home/wangjk/project/pyramid/data/public_originPic_aug"
    create_dir(new_data_path)
    """ Applying data augmentation. """
    augment_data(images, masks, new_data_path, augment=True)

    """ Loading augmented images and masks. """
    images, masks = load_data(new_data_path)
    print(f"Augmented Images: {len(images)} - Augmented Masks: {len(masks)}")