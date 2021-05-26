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
import ruihua
import args

# 输入：指定图像路径、mask和label的分辨率个数、是否数据增强、增强概率
# 输出：img的tensor，label和mask的tensor数组（5个分辨率），他们都是c*h*w,三通道，值为0 或者 1.0
class MSUDenseLeavesDataset(Dataset):

    # 初始化文件名路径和文件名列表
    def __init__(self, filepath, num_targets, random_augmentation=False, augm_probability=0.2):  # num_targets=5,
        self.filepath = filepath
        # each sample is made of image-labels-mask (important to sort by name!)
        self.images = sorted(glob.glob(filepath + '/*_img.png'))
        self.labels = sorted(glob.glob(filepath + '/*_label.png'))
        self.masks = sorted(glob.glob(filepath + '/*_mask.png'))
        self.n_samples = len(self.images)
        print("filepath + '*_img.png':", filepath + '/*_img.png', "len(self.images):", self.n_samples)
        self.multiscale_loss_targets = num_targets
        self.augmentation = random_augmentation
        self.probability = augm_probability

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):  # item这个下标对应的image，label，mask，返回的就是图片，只不过格式好像是tensor的
        # read image-labels-mask and return them
        image = cv2.imread(self.images[item])
        # 测试增加灰度处理，转为单通道
        if args.is_ruihua:
            imggray = cv2.imread(self.images[item], 0)
            imgLap = cv2.Laplacian(imggray, cv2.CV_16S)
            imgLapDelta = cv2.convertScaleAbs(imgLap)

            # Laplace Add
            imgLapAdd = cv2.addWeighted(imggray, 1.0, imgLap, -1.0, 0, dtype=cv2.CV_32F)
            imgLapAdd = cv2.convertScaleAbs(imgLapAdd)
            # print(image.shape)
            imgLapAdd = cv2.cvtColor(imgLapAdd, cv2.COLOR_GRAY2RGB)
            image = torch.from_numpy(imgLapAdd.transpose(2, 0, 1) / 255.).float()
            # 测试结束
        # HxWxC-->CxHxW, bgr->rgb and tensor transformation opencv读到的是hwc
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.transpose(2, 0, 1) / 255.).float()
        label = cv2.imread(self.labels[item])
        mask = cv2.imread(self.masks[item])


        # todo bug in generating dataset made following map have 3 (identical)
        #  channels instead of 1，改成 最大类间方差法
        label = label[:, :, 0] / 255.
        mask = mask[:, :, 0] / 255.
        # label = torch.from_numpy(label).long()
        # mask = torch.from_numpy(mask).float()

        # augmentation
        if self.augmentation:
            # augment data randomly
            if random.random() > (1 - self.probability):  # random.random()随机生成的一个实数，它在[0,1)范围内。
                # rotate
                angle = random.randint(0, 90)
                flip_rnd = random.random()

                # print('Rotating about', angle,'angle and additionally flipping with probability', flip_rnd)
                def rotate(img, angle):
                    img = TF.to_pil_image(img)  # torchvision.transforms.functional.to_pil_image(img)转换成PIL格式图像
                    img = TF.rotate(img, angle)
                    # additionally flip image with some prob
                    trans = []
                    if flip_rnd > (1 - 0.2):
                        trans.append(transforms.RandomVerticalFlip(1.0))  # 以给定的概率垂直翻转给定的图像。
                    if flip_rnd > (1 - 0.4):
                        trans.append(transforms.RandomHorizontalFlip(1.0))
                    trans.append(transforms.RandomAffine(degrees=30, translate=(0, 0.2)))
                    trans.append(transforms.ToTensor())
                    flip = transforms.Compose(trans)

                    return flip(img)

                image = rotate(image, angle)
                label = rotate(torch.from_numpy(label).float(),
                               angle).squeeze().numpy()  # 作用：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
                mask = rotate(torch.from_numpy(mask).float(), angle).squeeze().numpy()# todo:这里label和mask变成二值图了？

        # print((label.shape, mask.shape), (label.dtype, mask.dtype))
        # labels multiscale resizing
        targets, masks = multiscale_target(self.multiscale_loss_targets, label, mask)
        # reverse order of multiscale labels (long cast for CE loss)
        return image, [torch.from_numpy(t).unsqueeze(0) for t in reversed(targets)], [torch.from_numpy(m).unsqueeze(0)
                                                                                      for m in reversed(
                masks)]  # unsqueeze在指定维度进行维度扩充,reversed反转之后应该是从低分辨率到高分辨率吧


# normal multiscale does not achieve same results
# def multiscale(n_scaling, target, mask):
#     target = target.astype(np.float32)
#     mask = mask.astype(np.float32)
#     targets = [target]
#     masks = [mask]
#     parent_t = target
#     parent_mask = mask
#     for t in range(n_scaling):
#         scaled_t = cv2.resize(parent_t, (int(parent_t.shape[0]/2), int(parent_t.shape[1]/2)), interpolation=cv2.INTER_NEAREST).astype(np.float32)
#         scaled_m = cv2.resize(parent_mask, (int(parent_t.shape[0]/2), int(parent_t.shape[1]/2)), interpolation=cv2.INTER_NEAREST).astype(np.float32)
#         targets.append(scaled_t)
#         masks.append(scaled_m)
#
#         parent_t = scaled_t
#         parent_mask = scaled_m
#     return targets, masks

@njit  # 使用numba加速，对处理mask和target得到不同分辨率的label
def multiscale_target(n_targets, target, mask):  # target和mask是两维的numpy数组，值为0~1.0
    # targets = np.empty((self.multiscale_loss_targets, target.shape[0], target.shape[1]))
    targets = [target.astype(np.float32)]
    masks = [mask.astype(np.float32)]
    # outputs as many targets as the number of evaluations in the multiscale loss
    parent_target = target.astype(np.float32).copy()  # remember to uniform to same type(float32) or numba will complain
    parent_mask = mask.astype(np.float32).copy()
    for t in range(n_targets - 1):
        scaled_target = np.zeros((int(parent_target.shape[0] / 2), int(parent_target.shape[1] / 2))).astype(np.float32)
        scaled_mask = np.zeros((int(parent_target.shape[0] / 2), int(parent_target.shape[1] / 2))).astype(np.float32)
        for y, i in enumerate(range(1, parent_target.shape[0] - 1, 2)):  # 步长是2
            for x, j in enumerate(range(1, parent_target.shape[1] - 1, 2)):
                # check neighbour pixels for edges (clockwise check order) 检查相邻像素的边缘(顺时针检查顺序) 十字的四个角有一个是边就定义为边
                if parent_target[i, j - 1] == 1.0 or parent_target[i - 1, j] == 1.0 or \
                        parent_target[i, j + 1] == 1.0 or parent_target[i + 1, j] == 1.0:
                    scaled_target[y, x] = 1.0
                    scaled_mask[y, x] = 1.0
                # else if any of its parents are unknown it is unknown,  任何一个未知，就定义为未知,标记的是除边以外的所有像素
                elif parent_target[i, j - 1] == 0.0 or parent_target[i - 1, j] == 0.0 or \
                        parent_target[i, j + 1] == 0.0 or parent_target[i + 1, j] == 0.0:
                    scaled_target[y, x] = 0.0
                    scaled_mask[y, x] = 0.0
                # mask需要标记叶子内部的像素为1
                if parent_mask[i, j - 1] == 1.0 or parent_mask[i - 1, j] == 1.0 or \
                        parent_mask[i, j + 1] == 1.0 or parent_mask[i + 1, j] == 1.0:
                    # interior pixel
                    scaled_mask[y, x] = 1.0

        targets.append(scaled_target)
        masks.append(scaled_mask)
        parent_target = scaled_target
        parent_mask = scaled_mask
    # targets = [targets[-1],targets[-1],targets[-1],targets[-1],targets[-1]]
    # masks = [masks[0],masks[0],masks[0],masks[0],masks[0]]
    return targets, masks


if __name__ == '__main__':
    dataset = MSUDenseLeavesDataset('/home/wangjk/project/pyramid/data/gen/own_train224/', num_targets=5,
                                    random_augmentation=True,
                                    augm_probability=1.0)
    dataloader = DataLoader(dataset, batch_size=24)

    print(len(dataset))
    img, l, m = dataset[1] #l: torch.Size([1, 16, 16]) CxHxW
    print(img.shape)  # , l.shape, m.shape)   torch.Size([3, 256, 256])
    img = img.permute(1, 2, 0).numpy() * 255  # permute改变参数顺序
    img = img.astype(np.uint8)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    # cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    for target, mask in zip(l, m): # 分辨率由低到高
        target = target.squeeze().numpy() #shape:（16，16）
        mask = mask.squeeze().numpy() #(16, 16)
        print(target.shape, mask.shape)
        plt.imshow(target)
        plt.show()
        plt.imshow(mask)
        plt.show()
        # cv2.imshow()在服务器上不能正常运行
        # cv2.imshow('imga', target)
        # cv2.imshow('imgb', mask)
        # cv2.waitKey(0)
    # cv2.imshow('labels', l.numpy().astype(np.uint8)*255)
    # cv2.waitKey(0)
    # cv2.imshow('mask',  m.numpy().astype(np.uint8)*255)
    # cv2.waitKey(0)
