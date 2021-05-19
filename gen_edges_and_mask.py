import shutil
from argparse import ArgumentParser
from shutil import copy
import cv2
import numpy as np
import sys
import os

import random
from numba import jit
import glob
import tqdm

#
@jit(nopython=True)
def generate_edges_and_mask(image):
    # weird way to process it but this is how it was intended (took a while to figure it out..) 处理的方式很奇怪，但本来就是这样的
    segm = np.empty((image.shape[0], image.shape[1])).astype(np.int32)
    segm = image[:, :, 0] + 256 * image[:, :, 1] + 256 * 256 * image[:, :, 2]  # 为了确定边缘，把rgb三个通道的值乘起来之后再比较，不同的则是边界
    new_segm = np.zeros(segm.shape)
    mask = np.zeros(segm.shape)
    for i in range(1, segm.shape[0] - 1):
        for j in range(1, segm.shape[1] - 1):
            # spot contours 轮廓
            if segm[i - 1, j] != segm[i + 1, j] or segm[i, j - 1] != segm[i, j + 1] or \
                    segm[i + 1, j - 1] != segm[i + 1, j + 1] or segm[i - 1, j - 1] != segm[i - 1, j + 1]:  # todo:这么分不太合理吧，要么剪为4邻域，要么改成正宗的8邻域
                new_segm[i, j] = 1.0
                mask[i, j] = 1.0
            if segm[i, j] != 0:
                mask[i, j] = 1.0
    return new_segm, mask
path_testImgs="/home/wangjk/project/pyramid/testPic/labeled_pic"
list=glob.glob(f"{path_testImgs}/*_seg.png")
for imgpath in list:
    testImg = cv2.imread(imgpath)
    testImg_segm, testImg_mask = generate_edges_and_mask(testImg)
    path_testImg_segm = imgpath.replace('seg', 'label')
    path_testImg_mask = imgpath.replace('seg', 'mask')
    cv2.imwrite(path_testImg_segm, (testImg_segm * 255).astype(np.uint8))
    cv2.imwrite(path_testImg_mask, (testImg_mask * 255).astype(np.uint8))
# path_testImg="/home/wangjk/project/pyramid/testPic/DJI_0495_2_2_seg.png"
# path_testImg_segm="/home/wangjk/project/pyramid/testPic/DJI_0495_2_2_label.png"
# path_testImg_mask="/home/wangjk/project/pyramid/testPic/DJI_0495_2_2_mask.png"
# testImg=cv2.imread(imgpath)
# testImg_segm,testImg_mask=generate_edges_and_mask(testImg)
# cv2.imwrite(path_testImg_segm,(testImg_segm* 255).astype(np.uint8))
# cv2.imwrite(path_testImg_mask,(testImg_mask* 255).astype(np.uint8))
