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
# from PIL import Image
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
