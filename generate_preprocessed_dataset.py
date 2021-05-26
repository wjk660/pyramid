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

# 注意win下的目录是反斜杠，linux下是/，源代码是/的
def generate_patches_from_image(image, segm_map, mask, patch_size, filename, threshold=100): # patch_size:(128,128)
    """
    Returns a set of cropped patched from the provided image by tiling over the image with a window of size patch_size.
    Only patches containing at least 'threshold' labeled pixels will be returned.有不少于threshold 100个label像素的切片才能被返回
    """
    patch_no = 0
    filename = filename[:-8]  # remove "_seg.png"
    for y in range(patch_size[0], image.shape[0], patch_size[0]):  # 开始，结束，步长) 用于求出batch的分界点
        for x in range(patch_size[1], image.shape[1], patch_size[1]):
            # keep it simple, avoid padding todo:可以优化（允许重叠）
            mask_patch = mask[y - patch_size[0]:y, x - patch_size[1]:x]
            if np.count_nonzero(mask_patch) > threshold:
                # write to disk image-labels-mask
                image_patch = image[y - patch_size[0]:y, x - patch_size[1]:x, :]
                label_patch = segm_map[y - patch_size[0]:y, x - patch_size[1]:x]
                # print(f'Writing to: {filename}_{patch_no}_img.png')
                # print(image_patch.shape, mask_patch.shape, label_patch.shape)
                cv2.imwrite(f'{filename}_{patch_no}_img.png', image_patch)
                cv2.imwrite(f'{filename}_{patch_no}_mask.png', (mask_patch * 255).astype(np.uint8)) # 保存成单通道的二值图
                cv2.imwrite(f'{filename}_{patch_no}_label.png', (label_patch * 255).astype(np.uint8))
                patch_no += 1


# # generate binary edges and mask from instance segmentation
# @jit(nopython=True)
# def generate_edges_and_mask(image):
#     # weird way to process it but this is how it was intended (took a while to figure it out..) 处理的方式很奇怪，但本来就是这样的
#     segm = np.empty((image.shape[0], image.shape[1])).astype(np.int32)
#     segm = image[:, :, 0] + 256 * image[:, :, 1] + 256 * 256 * image[:, :, 2]  # 为了确定边缘，把rgb三个通道的值乘起来之后再比较，不同的则是边界
#     new_segm = np.zeros(segm.shape)
#     mask = np.zeros(segm.shape)
#     for i in range(1, segm.shape[0] - 1):
#         for j in range(1, segm.shape[1] - 1):
#             # spot contours 轮廓
#             if segm[i - 1, j] != segm[i + 1, j] or segm[i, j - 1] != segm[i, j + 1] or \
#                     segm[i + 1, j - 1] != segm[i + 1, j + 1] or segm[i - 1, j - 1] != segm[i - 1, j + 1]:  # todo:这么分不太合理吧，要么剪为4邻域，要么改成正宗的8邻域
#                 new_segm[i, j] = 1.0
#                 mask[i, j] = 1.0
#             if segm[i, j] != 0:
#                 mask[i, j] = 1.0
#     return new_segm, mask
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

def mainprocess(origin_dataset_path,target_dataset_path,patch_size):
    for filename in tqdm.tqdm(glob.glob(f'{origin_dataset_path}/*_seg.png')):  # tqdm用于显示进度条
        # print(filename, dataset, dirname, os.path.join(dataset, dirname))
        # print(filename)
        label = cv2.imread(filename)
        original_image = cv2.imread(filename.replace('seg', 'img'))  # 替换(old,new)

        edges_map, mask = generate_edges_and_mask(label)  # 用标签图像制作一个边缘图像，一个mask掩膜图像

        f = filename.split('/')[-1]
        # now split image in multiple patches (raw image augmentation)
        # print(args.dataset, dirname, f,"----")
        output_dir =os.path.join(target_dataset_path, f)
        # print(output_dir)
        generate_patches_from_image(original_image, edges_map, mask,patch_size, output_dir,
                                    threshold=threshold)

    # ax[2].imshow(cv2.imread(filenameim) / 255. + mask[:, :, None])
def divide_set():
    #  目标文件夹如果不存在，直接创建她
    if not os.path.exists(origin_dataset_path):
        raise ("not folder or no picture")
    for path in [divide_dataset_path_train, divide_dataset_path_val, divide_dataset_path_test, divide_dataset_path_all]:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)
    # 划分数据集
    list_ImgsPath = glob.glob(f'{origin_dataset_path}/*_seg.png')  # 读出所有文件夹的路径
    # images = os.listdir(origin_dataset_path)
    num = len(list_ImgsPath)
    train_index = random.sample(list_ImgsPath, k=int(num * split_rate[0]))
    for index, image in enumerate(list_ImgsPath):
        if image in train_index:  # 如果在训练集中
            # 移动seg.png
            image_path = image
            new_path = divide_dataset_path_train
            copy(image_path, new_path)  # shutil.copy()可将文件拷贝到指定文件夹
            # 移动img.png
            image_path = image.replace('seg', 'img')
            copy(image_path, new_path)  # shutil.copy()可将文件拷贝到指定文件夹
        else:  # val
            image_path = image
            new_path = divide_dataset_path_val
            copy(image_path, new_path)
            image_path = image.replace('seg', 'img')
            copy(image_path, new_path)  # shutil.copy()可将文件拷贝到指定文件夹
        #  all
        image_path = image
        new_path = divide_dataset_path_all
        copy(image_path, new_path)
        image_path = image.replace('seg', 'img')
        copy(image_path, new_path)  # shutil.copy()可将文件拷贝到指定文件夹
        print("\r processing [{}/{}]".format(index + 1, num), end="")  # processing bar

# def merge_dataset():
#     list_ImgsPath = glob.glob(f'{used_dataset_path_train}/*.png')  # 读出所有文件夹的路径

# 作用：划分train、val和test（暂无），之后切成小片
if __name__ == '__main__':
    patch_size = (224, 224)
    # 原始数据集
    dataset = "/home/wangjk/dataset/DenseLeaves/"
    dataset_subdir = ["own_instancePic/","public_originPic/"]#原始数据集
    # 划分数据集
    divide_rootdir="/home/wangjk/dataset/DenseLeaves/split/"  # 存放按比例分配后的数据
    divide_folder_names_own=["own_train/","own_val/","own_test/","own_all/"]
    divide_folder_names_public=["public_train/","public_val/","public_test/","public_all/"]
    divide_folder=[divide_folder_names_own,divide_folder_names_public]
    # 使用数据集
    used_dir = "/home/wangjk/dataset/DenseLeaves/gen/"
    uesd_folder_ownPic=[f"own_train{patch_size[0]}/", f"own_val{patch_size[0]}/", f"own_test{patch_size[0]}/",f"own_all{patch_size[0]}/"]
    uesd_folder_publicPic = [f"public_train{patch_size[0]}/", f"public_val{patch_size[0]}/",
                            f"public_test{patch_size[0]}/",f"public_all{patch_size[0]}/"]
    uesd_folder_OwnAndPublic=[f"public_and_own_train{patch_size[0]}/", f"public_and_own_val{patch_size[0]}/",f"public_and_own_test{patch_size[0]}/",f"public_and_own_all{patch_size[0]}/"]
    uesd_folders = [uesd_folder_ownPic, uesd_folder_publicPic, uesd_folder_OwnAndPublic]

    split_rate=[0.9,0.1,0,1] #train:val:test:all，目前只是用train、val、all
    # assume ~1/4 of image has to be labeled for patch threshold (in case of 128x128 patch size)
    threshold = 3800

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!重点两行!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 设置使用哪个原始还是公开的
    used_dataset_subdir=dataset_subdir[1]
    # 划分数据集是公开的还是自己的
    divide_folder_names=divide_folder[1]
    # 是否已经进行了divide划分操作
    has_divide = False
    # 使用自己的、公开的、还是两个都用 0：自己的，1：公开的，2：都用
    used_folder=uesd_folders[2]
    # # 是否将own和public合并到public_and_own_...中
    # is_merge=True
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!重点两行!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    origin_dataset_path = dataset + used_dataset_subdir  # 拼出原始数据集的路径
    divide_dataset_path_train = divide_rootdir + divide_folder_names[0]
    divide_dataset_path_val = divide_rootdir + divide_folder_names[1]
    divide_dataset_path_test = divide_rootdir + divide_folder_names[2]
    divide_dataset_path_all = divide_rootdir + divide_folder_names[3]

    used_dataset_path_train = used_dir + used_folder[0]  # 拼出裁剪之后数据集的路径
    used_dataset_path_val = used_dir + used_folder[1]  # 拼出裁剪之后数据集的路径
    used_dataset_path_test = used_dir + used_folder[2]  # 拼出裁剪之后数据集的路径
    used_dataset_path_all = used_dir + used_folder[3]  # 拼出裁剪之后数据集的路径

    # if is_merge:
    #     merge_dataset()
    if has_divide==False:
        divide_set()
    # 划分完成后，进行切片处理
    for path in [used_dataset_path_train,used_dataset_path_val,used_dataset_path_test,used_dataset_path_all]:
        if not os.path.exists(path):
            os.makedirs(path)
        else:# 执行代码前清空指定的文件夹
            shutil.rmtree(path)
            os.mkdir(path)
    mainprocess(divide_dataset_path_all, used_dataset_path_all, patch_size)
    mainprocess(divide_dataset_path_train, used_dataset_path_train, patch_size)
    mainprocess(divide_dataset_path_val, used_dataset_path_val, patch_size)


