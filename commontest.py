
import cv2
import numpy as np
import sys
import os
from numba import jit
import glob
import tqdm
import torch.nn as nn
import torch
from utils import device
from utils import parse_args
from msu_leaves_dataset import MSUDenseLeavesDataset
from torch.utils.data import DataLoader
from pyramid_network import PyramidNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

'''
matplotlib 目前不支持汉语，需要下载字体并进行管理
'''

#  ---------------------------测试绘图
import glob

'''
plt.legend（）函数主要的作用就是给图加上图例，plt.legend([x,y,z……])里面的参数使用的是list的的形式将图表的的名称喂给这个函数。
'''
# import matplotlib.pyplot as plt
#
# x = [1,2,3,4,5,6,7,8,]
# y1 = [1,2,3,4,55,6,6,7]
#
# y2 = [6,20,9,2,5,8,2,8,]
#
# plt.plot(x,y1)
# plt.plot(x,y2)
#
# plt.legend(['y1','y2'])
# plt.show()

# 替换命令行输入
# import os
# print(os.path.join(os.getcwd(),"kllj"))
# class args:
#     bs="jk"
#     aa="sf"
# print(args.bs)

# import matplotlib as plt
# img=Image.open('./test.JPG')
# plt.figure()
# plt.imshow(img)
# plt.show()
# plt()
# print(os.getcwd())
# lis=[1,2,3,4,5]
# print(lis[:3])
# print(lis[:-3])

# -------------测试enumerate
# list1 = ["这", "是", "一个", "测试"]
# for index, item in enumerate(list1[1:]):
#     print(index, item)
# 测试[ print() ]
# [print(y) for y in [1,2]]

# # 测试glob
# filepath = '/home/wangjk/dataset/DenseLeaves/train/'
# # each sample is made of image-labels-mask (important to sort by name!)
# images = glob.glob(filepath + '*_img.png')
# labels = sorted(glob.glob(filepath + '*_label.png'))
# masks = sorted(glob.glob(filepath + '*_mask.png'))
# print(len(images))

# 测试
import torch
print(torch.cuda.is_available())
# #测试:的优先级,切片的优先级比+、-等运算符低
# list1 = ["这", "是", "一个", "测试"]
# #out:['是', '一个', '测试']
# print(list1[(3-2):4])
# print(list1[(3-2):4])

# # 测试：segm = image[:, :, 0] + 256 * image[:, :, 1] + 256 * 256 * image[:, :, 2]的效果
# import numpy as np
# import cv2
#
# class args:
#     filename = "/home/wangjk/dataset/DenseLeaves/train/leaf00153_045_seg.png"
#     filename2="/home/wangjk/project/pyramid/testPic/DJI_0986.JPG"
#     subdir = "train"
#     patch_size = (128, 128)
#     output_dir = "/home/wangjk/dataset/DenseLeaves/gen/train"
# a = np.array([[1,  2],  [3,  4]])
# b = np.array([[1,  2],  [3,  4]])
# print (a,a.shape,a.dtype) # [[1 2], [3 4]]  (2, 2) int64
# print(a[:,0]+3*b[:,1])  # [ 7 15]
# image = cv2.imread(args.filename2)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # cv2.imshow('image',image)
#
# plt.imshow(image)
# plt.show()
# img=Image.open(args.filename2)
# plt.imshow(img)
# plt.show()
#
# segm = image[:, :, 0] + 256 * image[:, :, 1] + 256 * 256 * image[:, :, 2]  # ??为什么？
# plt.imshow(segm,)
# plt.show()
# 测试基本语法
p=torch.rand([2, 3])
print(p)
print((p<0.5).float())
