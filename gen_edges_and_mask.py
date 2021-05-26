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

# 利用标注好的图片生成边缘（真正的label），返回的是浮点二值数组（0或者1.0）
@jit(nopython=True)
def generate_edges_and_mask(image):
    # weird way to process it but this is how it was intended (took a while to figure it out..) 处理的方式很奇怪，但本来就是这样的
    segm = np.empty((image.shape[0], image.shape[1])).astype(np.int32)
    segm = image[:, :, 0] + 256 * image[:, :, 1] + 256 * 256 * image[:, :, 2]  # 为了确定边缘，把rgb三个通道的值乘起来之后再比较，不同的则是边界
    new_segm = np.zeros(segm.shape)
    mask = np.zeros(segm.shape)
    for i in range(0, segm.shape[0] ):
        for j in range(0, segm.shape[1] ):
            # spot contours 轮廓
            if i==0 or i==segm.shape[0] or j==0 or j==segm.shape[1]: # 添加了对于边缘像素的处理
                if segm[i,j]!=0:
                    new_segm[i, j] = 1.0
                    mask[i, j] = 1.0
                continue
            if segm[i - 1, j] != segm[i + 1, j] or segm[i, j - 1] != segm[i, j + 1] or \
                    segm[i + 1, j - 1] != segm[i - 1, j + 1] or segm[i - 1, j - 1] != segm[i + 1, j - 1]:  # now:改成正宗的8邻域
                new_segm[i, j] = 1.0
                # mask[i, j] = 1.0
            if segm[i, j] != 0:
                mask[i, j] = 1.0
    return new_segm, mask # 返回的边缘和掩码是两维矩阵，值浮点值是0或者1.0，不是整形。

if __name__=="__main__":
    path_testImgs="/home/wangjk/project/pyramid/testPic/labeled_pic" # 存放标注图片的文件夹
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
