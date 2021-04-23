import os
import cv2
import torch
from utils import device
from utils import parse_args
from pyramid_network import PyramidNet
import numpy as np
import matplotlib.pyplot as plt
import faulthandler
from logger import Logger
import sys
import train

import glob
import os
from os.path import isdir
import shutil
from unet_model import UNet
# 调试注意：更改origin_path和bin_path、args.loadModelName
origin_path="/home/wangjk/dataset/DenseLeaves/classification/origin"
bin_path="/home/wangjk/dataset/DenseLeaves/classification/bin/unet/epoch2000"
if not os.path.exists(origin_path):
    raise("not folder or no picture")
if not os.path.exists(bin_path):
    os.mkdir(bin_path)
# 执行代码前清空指定的文件夹
shutil.rmtree(bin_path)
os.mkdir(bin_path)
#改名字
# for jsonfile in origin_folders_paths:
class args:
    loadModelName="Epoch2000Pathsize256BilinearUnet"
    load_model = f'/home/wangjk/project/pyramid/modelParameter/{loadModelName}.pt'
    pic_name="测试专用.JPG"
    pic_prefix=pic_name.split('.')[0]
    image = f"/home/wangjk/project/pyramid/testPic/{pic_name}"
    save_path = ""
describe = f"epoch{train.args.epochs},patch_size = {train.args.patch_size}*{train.args.patch_size},add three pic that labeled all leaves"

if __name__ == '__main__':
    sys.stdout = Logger("log/custom_image.txt")
    print(describe)
    faulthandler.enable()  # 报错时能看到原因
    # args = parse_args()
    # args={"load_model":'pyramid_net_model.pt',"image":'DJI_0986.JPG',"save_path":""}

    # todo totally arbitrary weights
    # model = PyramidNet(n_layers=5, loss_weights=[torch.tensor([1.0])]*5)#, torch.tensor([1.9]), torch.tensor([3.9]),
    model = UNet(n_channels=3, n_classes=1, bilinear=False)  # , torch.tensor([1.1]), torch.tensor([1.8]),
    testRes = f"{args.image.split('/')[-1].split('.')[0]}_By_{args.load_model.split('/')[-1].split('.')[0]}"
    # torch.tensor([8]), torch.tensor([10])])
    # print(model.parameters())
    device=torch.device("cuda:2")
    model = model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    else:
        print("Please provide a valid path to the pre-trained model to evaluate")
        exit(1)


    list_imgsPath = [image_test for image_test in glob.glob(origin_path + "/*.JPG")]  # 读出所有文件夹的路径
    for i in range(len(list_imgsPath)):
        og_image = cv2.imread(list_imgsPath[i])
        # if og_image.shape[0] > 1024 and og_image.shape[1]>1024:  # 从中间取1024*1024大小的图像
        #     og_image = og_image[(og_image.shape[0]-1024)//2:og_image.shape[0]-(og_image.shape[0]-1024)//2, (og_image.shape[1]-1024)//2:og_image.shape[1]-(og_image.shape[1]-1024)//2, :]
        # resize image or crop patches
        # image = cv2.resize(og_image, (128, 128))
        # og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
        # image = torch.from_numpy(og_image.transpose(2, 0, 1) / 255.).float()

        # 增加灰度处理，转为单通道
        # imgname = og_image
        imggray = cv2.imread(list_imgsPath[i],0)
        # plt.imshow(imggray, cmap='gray')
        # plt.show()
        imgLap = cv2.Laplacian(imggray, cv2.CV_16S)
        # plt.imshow(imgLap)
        # plt.show()
        imgLapDelta = cv2.convertScaleAbs(imgLap)
        # plt.imshow(imgLapDelta)
        # plt.show()
        # Laplace Add
        imgLapAdd = cv2.addWeighted(imggray, 1.0, imgLap, -1.0, 0, dtype=cv2.CV_32F)
        imgLapAdd = cv2.convertScaleAbs(imgLapAdd)
        # print(image.shape)
        imgLapAdd = cv2.cvtColor(imgLapAdd, cv2.COLOR_GRAY2RGB)
        image = torch.from_numpy(imgLapAdd.transpose(2, 0, 1) / 255.).float()

        image = image.to(device)
        model.eval()  # 设置为evaluation模式
        with torch.no_grad():
            predictions = model(image.unsqueeze(0))
            # get prediction at max resolution predictions本身返回的是多个分辨率都有的，现在只保留最高的分辨率那个，因为实验表明这个最好
            p = predictions[-1]
            # sigmoid + thresholding
            p = (p > 0.).float() # 如果大于0，则置为1
            p = p.squeeze().cpu().numpy().astype(np.float32)  # p变成2维的了,值是 0或者1
            Name_binImg=list_imgsPath[i].split('/')[-1]
            cv2.imwrite(os.path.join(bin_path,f'bin_{Name_binImg}' ), (p * 255).astype(np.uint8))
            # cv2.imwrite(os.path.join(bin_path, f'bin_{args.pic_prefix}_{args.loadModelName}.png'), (p * 255).astype(np.uint8))
            # cv2.imwrite(os.path.join(args.save_path, f'testPic/mix_{args.pic_prefix}_{train.saveModelName}.png'), (p.unsqueeze(0) * 255+og_image).astype(np.uint8))
            # res=cv2.cvtColor(p,)
            # mix = cv2.addWeighted(og_image, 0.7, (p * 255).astype(np.uint8), 0.3, 0)
            # cv2.imshow('mix', mix)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(p.shape)
            # plt.figure()
            # plt.subplot(2,2,1)
            # plt.imshow(p)
            # plt.subplot(2,2,2)
            # # fix, ax = plt.subplots(1, 2)
            #
            # # ax[0].imshow(og_image)
            # # ax[1].imshow(p, cmap='Greys')
            #
            # # plt.legend(['ax[0]','ax[1]'])
            # plt.imshow(og_image)
            # plt.subplot(2,2,3)
            # 为了生成用于训练的图像把这段注释掉了
            # plt.imshow(og_image,alpha=0.7)
            # plt.imshow(p,alpha=0.3)
            # plt.savefig(f"/home/wangjk/project/pyramid/testPic/mix_{args.pic_prefix}_{args.loadModelName}.png")
            # plt.show()



