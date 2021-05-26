import os
import sys
import testPic
# import torch.nn as nn
import time
import cv2
import torch
def ruihua(path_img,channel_num):
    if channel_num==1:
        imggray = cv2.imread(path_img, 0)
        imgLap = cv2.Laplacian(imggray, cv2.CV_16S)# cv2.CV_16S使用16位有符号的数据类型,否则有截断https://blog.csdn.net/sunny2038/article/details/9170013
        imgLapDelta = cv2.convertScaleAbs(imgLap)
        # Laplace Add
        imgLapAdd = cv2.addWeighted(imggray, 1.0, imgLap, -1.0, 0, dtype=cv2.CV_32F)
        imgLapAdd = cv2.convertScaleAbs(imgLapAdd)
        # print(image.shape)
        imgLapAdd = cv2.cvtColor(imgLapAdd, cv2.COLOR_GRAY2RGB)
        image = torch.from_numpy(imgLapAdd.transpose(2, 0, 1) / 255.).float()
    elif channel_num==3:
        imggray = cv2.imread(path_img)

        imgLap = cv2.Laplacian(imggray, cv2.CV_16S)

        imgLapDelta = cv2.convertScaleAbs(imgLap)

        # Laplace Add
        imgLapAdd = cv2.addWeighted(imggray, 1.0, imgLap, -1.0, 0, dtype=cv2.CV_32F)
        imgLapAdd = cv2.convertScaleAbs(imgLapAdd)
        # print(image.shape)
        imgLapAdd = cv2.cvtColor(imgLapAdd, cv2.COLOR_GRAY2RGB)
        image = torch.from_numpy(imgLapAdd.transpose(2, 0, 1) / 255.).float()
