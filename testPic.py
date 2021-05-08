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
import ruihua
import args

def test_picture(dirpath_orginpics,bin_outputPath,model,path_load_model,name_load_model):
    if model:
        print("has model")
        print(dirpath_orginpics,bin_outputPath,path_load_model,name_load_model)
    else:
        print("Please provide a valid model to evaluate")
        exit(1)
    if path_load_model:
        model.load_state_dict(torch.load(path_load_model))
    else:
        print("Please provide a valid path to evaluate")
        exit(1)
    model.eval()  # 设置为evaluation模式
    list_imgsPath = [image_test for image_test in glob.glob(dirpath_orginpics + "/*.JPG")]  # 读出所有文件夹的路径
    for i in range(len(list_imgsPath)):
        print("dingwei1")
        image = cv2.imread(list_imgsPath[i])
        if args.is_ruihua:
            # 增加灰度处理，转为单通道
            # imgname = og_image
            imggray = cv2.imread(list_imgsPath[i],0)
            imgLap = cv2.Laplacian(imggray, cv2.CV_16S)
            imgLapDelta = cv2.convertScaleAbs(imgLap)
            # Laplace Add
            imgLapAdd = cv2.addWeighted(imggray, 1.0, imgLap, -1.0, 0, dtype=cv2.CV_32F)
            imgLapAdd = cv2.convertScaleAbs(imgLapAdd)
            # print(image.shape)
            imgLapAdd = cv2.cvtColor(imgLapAdd, cv2.COLOR_GRAY2RGB)
            image = torch.from_numpy(imgLapAdd.transpose(2, 0, 1) / 255.).float()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_opencv=image.copy()
        image = torch.from_numpy(image.transpose(2, 0, 1) / 255.).float()
        image = image.to(device)
        print("dingwei2")
        with torch.no_grad():
            print("dingwei3")
            predictions = model(image.unsqueeze(0))
            # get prediction at max resolution predictions本身返回的是多个分辨率都有的，现在只保留最高的分辨率那个，因为实验表明这个最好
            p = predictions[-1]
            # sigmoid + thresholding
            p = (p > 0.).float() # 如果大于0，则置为1
            p = p.squeeze().cpu().numpy().astype(np.float32)  # p变成2维的了,值是 0或者1
            Name_binImg=list_imgsPath[i].split('/')[-1]
            res_binImg=(p * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(bin_outputPath,f'{name_load_model}-{Name_binImg}' ), res_binImg)
            mask_inv=cv2.bitwise_not(res_binImg)
            res_mixImg = cv2.bitwise_and(image_opencv,image_opencv, mask=mask_inv)
            cv2.imwrite(os.path.join(bin_outputPath,f'{name_load_model}-mix-{Name_binImg}' ), res_mixImg)
            print("dingwei4")
    print("finish testPic sucessfully!")
    exit(0)
if __name__ == '__main__':
    # sys.stdout = Logger("log/custom_image.txt")
    # print(describe)
    faulthandler.enable()  # 报错时能看到原因
    # 调试注意：更改origin_path和bin_path、args.loadModelName
    origin_path = "/home/wangjk/dataset/DenseLeaves/classification/origin"
    bin_path = "/home/wangjk/dataset/DenseLeaves/classification/bin"
    if not os.path.exists(origin_path):
        raise ("not folder or no picture")
    if not os.path.exists(bin_path):
        os.mkdir(bin_path)
    # 执行代码前清空指定的文件夹
    # shutil.rmtree(bin_path)
    # os.mkdir(bin_path)

    # 改名字
    # for jsonfile in origin_folders_paths:
    loadModelName = "05-05_15-06Size224Best"
    load_model = f'/home/wangjk/project/pyramid/modelParameter/{loadModelName}.pt'
    # pic_name = "测试专用.JPG"
    # pic_prefix = pic_name.split('.')[0]
    # image = f"/home/wangjk/project/pyramid/testPic/{pic_name}"
    # save_path = ""
    describe = f"epoch{train.args.epochs},patch_size = {train.args.patch_size}*{train.args.patch_size},add three pic that labeled all leaves"

    # todo totally arbitrary weights
    # model = PyramidNet(n_layers=5, loss_weights=[torch.tensor([1.0])]*5)#, torch.tensor([1.9]), torch.tensor([3.9]),
    model = PyramidNet(n_layers=5, loss_weights=[torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]),
                                                 torch.tensor([1.0]), torch.tensor([0.1])])#
    device=torch.device("cuda:1")
    model = model.to(device)
    test_picture(origin_path,bin_path,model,load_model,loadModelName)






