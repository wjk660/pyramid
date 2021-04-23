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
                cv2.imwrite(f'{filename}_{patch_no}_mask.png', (mask_patch * 255).astype(np.uint8))
                cv2.imwrite(f'{filename}_{patch_no}_label.png', (label_patch * 255).astype(np.uint8))
                patch_no += 1


# generate binary edges and mask from instance segmentation
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

def mainprocess(origin_dataset_path,target_dataset_path,patch_size):
    for filename in tqdm.tqdm(glob.glob(f'{origin_dataset_path}/*_seg.png')):  # tqdm用于显示进度条
        # print(filename, dataset, dirname, os.path.join(dataset, dirname))
        # print(filename)
        label = cv2.imread(filename)
        original_image = cv2.imread(filename.replace('seg', 'img'))  # 替换(old,new)
        # cv2.imshow('a',label)
        # cv2.waitKey(0)
        # cv2.imshow('b',original_image)
        # cv2.waitKey(0)
        # exit(0)
        # generate edge labels and mask from instance segmentation image
        edges_map, mask = generate_edges_and_mask(label)  # 用标签图像制作一个边缘图像，一个mask掩膜图像

        f = filename.split('/')[-1]
        # now split image in multiple patches (raw image augmentation)
        # print(args.dataset, dirname, f,"----")
        output_dir =os.path.join(target_dataset_path, f)
        # print(output_dir)
        generate_patches_from_image(original_image, edges_map, mask,patch_size, output_dir,
                                    threshold=threshold)

    # ax[2].imshow(cv2.imread(filenameim) / 255. + mask[:, :, None])

if __name__ == '__main__':
    class args:
        dataset = "/home/wangjk/dataset/DenseLeaves/"
        dataset_subdir = ["own_instancePic/","public_originPic/"]#原始数据集
        patch_size = (256, 256)
        split_dir="/home/wangjk/dataset/DenseLeaves/split/"
        output_dir = "/home/wangjk/dataset/DenseLeaves/gen/"
        output_subdir_ownPic=[f"own_instance_train{patch_size[0]}/", f"own_instance_val{patch_size[0]}/", f"own_instance_test{patch_size[0]}/",f"own_instance_all{patch_size[0]}/"]
        output_subdir_publicPic = [f"public_instance_train{patch_size[0]}/", f"public_instance_val{patch_size[0]}/",
                                f"public_instance_test{patch_size[0]}/",f"public_instance_all{patch_size[0]}/"]
        output_subdir_OwnAndPublic=[f"public_and_own_instance_train{patch_size[0]}/", f"public_and_own_instance_val{patch_size[0]}/",f"public_and_own_instance_test{patch_size[0]}/",f"public_and_own_instance_all{patch_size[0]}/"]
        choose_which_set="only_own"#"only_own","only_public","own_public"
        split_rate=[0.9,0.1,0,1] #train:val:test:all，目前只是用train、val、all


    # assume ~1/4 of image has to be labeled for patch threshold (in case of 128x128 patch size)
    threshold = 3800
    # print(args.subdir, args.dataset)
    # 循环处理没有旋转的图片
    # for filename in tqdm.tqdm(glob.glob(args.dataset + f'/{args.subdir}/*000_seg.png')): # tqdm用于显示进度条
    if args.choose_which_set == "only_own":
        origin_dataset_path = args.dataset + args.dataset_subdir[0]  # 拼出原始数据集的路径
        split_dataset_path_train=args.split_dir+args.output_subdir_ownPic[0]
        split_dataset_path_val=args.split_dir+args.output_subdir_ownPic[1]
        split_dataset_path_test=args.split_dir+args.output_subdir_ownPic[2]
        split_dataset_path_all=args.split_dir+args.output_subdir_ownPic[3]
        target_dataset_path_train = args.output_dir + args.output_subdir_ownPic[0]  # 拼出裁剪之后数据集的路径
        target_dataset_path_val = args.output_dir + args.output_subdir_ownPic[1]  # 拼出裁剪之后数据集的路径
        target_dataset_path_test = args.output_dir + args.output_subdir_ownPic[2]  # 拼出裁剪之后数据集的路径
        target_dataset_path_all = args.output_dir + args.output_subdir_ownPic[3]  # 拼出裁剪之后数据集的路径
        #  目标文件夹如果不存在，直接创建她
        if not os.path.exists(origin_dataset_path):
            raise("not folder or no picture")
        for path in [split_dataset_path_train,split_dataset_path_val,split_dataset_path_test,split_dataset_path_all]:
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                shutil.rmtree(path)
                os.mkdir(path)
        for path in [target_dataset_path_train,target_dataset_path_val,target_dataset_path_test,target_dataset_path_all]:
            if not os.path.exists(path):
                os.makedirs(path)
            else:# 执行代码前清空指定的文件夹
                shutil.rmtree(path)
                os.mkdir(path)

        # 划分数据集
        list_ImgsPath = [imagePath for imagePath in glob.glob(f'{origin_dataset_path}/*_seg.png')]  # 读出所有文件夹的路径
        # images = os.listdir(origin_dataset_path)
        num = len(list_ImgsPath)
        train_index = random.sample(list_ImgsPath, k=int(num * args.split_rate[0]))
        for index, image in enumerate(list_ImgsPath):
            if image in train_index:  # 如果在训练集中
                # 移动seg.png
                image_path = image
                new_path = split_dataset_path_train
                copy(image_path, new_path) #shutil.copy()可将文件拷贝到指定文件夹
                #移动img.png
                image_path = image.replace('seg', 'img')
                copy(image_path, new_path) #shutil.copy()可将文件拷贝到指定文件夹
            else:
                image_path = image
                new_path = split_dataset_path_val
                copy(image_path, new_path)
                #移动img.png
                image_path = image.replace('seg', 'img')
                copy(image_path, new_path) #shutil.copy()可将文件拷贝到指定文件夹
            image_path = image
            new_path = split_dataset_path_all
            copy(image_path, new_path)
            # 移动img.png
            image_path = image.replace('seg', 'img')
            copy(image_path, new_path)  # shutil.copy()可将文件拷贝到指定文件夹
            print("\r processing [{}/{}]".format(index + 1, num), end="")  # processing bar

        mainprocess(split_dataset_path_all, target_dataset_path_all, args.patch_size)
        mainprocess(split_dataset_path_train, target_dataset_path_train, args.patch_size)
        mainprocess(split_dataset_path_val, target_dataset_path_val, args.patch_size)

    # origin_dataset_path=args.dataset+args.dataset_subdir[1]  # 拼出原始数据集的路径
    # target_dataset_path=args.output_dir+args.output_subdir_ownPic[1] # 拼出裁剪之后数据集的路径
    # # realUsed_dataset_path=args.output_dir+args.subdir_realUsed[1] # 拼出裁剪之后数据集的路径
    # if not os.path.exists(origin_dataset_path):
    #     os.makedirs(origin_dataset_path)
    # if not os.path.exists(target_dataset_path):
    #     os.makedirs(target_dataset_path)
    # # 执行代码前清空指定的文件夹
    # shutil.rmtree(target_dataset_path)
    # os.mkdir(target_dataset_path)
    # mainprocess(origin_dataset_path, target_dataset_path, args.patch_size)
    # # shutil.move(target_dataset_path,realUsed_dataset_path)


