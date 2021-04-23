import os
import sys

# import torch.nn as nn
import time

import torch

from logger import Logger
from utils import device
from utils import parse_args
from msu_leaves_dataset import MSUDenseLeavesDataset
from torch.utils.data import DataLoader
from pyramid_network import PyramidNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from unet_model import UNet

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now()) # 添加时间戳解决，曲线显示紊乱的问题
def evaluate(net, eval_dataset):
    net.eval()
    print('*'*50)
    count = 0
    sum_acc = 0
    with torch.no_grad():
        for batch_no, (image, targets, masks) in tqdm(enumerate(eval_dataset)):
            image = image.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            predictions = net(image)
            loss = compute_loss(predictions, targets[-1], masks[-1])
            # for i in range(len(predictions)):
            #  print(predictions[i].shape, targets[i].shape, masks[i].shape)
            print('Eval Loss:', loss.item())
            # pixel-wise accuracy of multiscale predictions (edges-only)

            for p, t, m in zip(predictions, targets[-1], masks[-1]):# predictions是模型的输出，值是0.265这种，targets和 masks是标签和掩码，值是0或1

                p = (p>0.).float()
                pixel_acc = (p * m) * t #先用掩码做与操作，因为只关注标注了的部分，只有这部分才有label，然后和targets做与，看看有没有标全
                acc = pixel_acc.sum() / t.sum()
                print(f"Accuracy at scale in valset({p.shape[1]}x{p.shape[2]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")
                count = count + 1
                sum_acc=sum_acc+acc
    # writer_acc_val.add_scalar('acc',
    #   sum_acc/count,
    #   global_step=epoch)
    writer_loss_val.add_scalar('loss',
                      loss.item(),
                      global_step=epoch)
    print('*'*50)
    net.train()

def compute_loss(prediction, target, mask):
    # reduction logic: mask is applied afterwards on the non-reduced loss
    losses = [torch.sum(criterion(prediction, target) * mask) / torch.sum(mask) ]# todo：为什么这么计算loss
    # here sum will call overridden + operator
    return sum(losses)
# 注意更改：epochs和saveModelName的值,train_prefix,val_prefix
class args:
    dataset_filepath = "/home/wangjk/dataset/DenseLeaves/gen/"
    epochs = 2000
    seed = 7
    log_interval = 10
    predictions_number = 5
    save_path = "/home/wangjk/project/pyramid/modelParameter/"
    # load_model = "pyramid_net_model.pt"
    load_model = False
    viz_results = False
    patch_size = 256
    train_prefix="own_instance_all" #训练集所在文件夹名称的前缀 训练集=前缀+patch_size
    val_prefix="own_instance_val"

# args = parse_args()
saveModelName = f"Epoch{args.epochs}Pathsize{args.patch_size}BilinearUnet" #保存模型名称的前缀
describe = f"epoch{args.epochs},patch_size = {args.patch_size}*{args.patch_size},add three pic that labeled all leaves"
# 最终使用的训练集和验证集路径
trainset_path= args.dataset_filepath + f'{args.train_prefix}{args.patch_size}'
valset_path= args.dataset_filepath + f'{args.val_prefix}{args.patch_size}'
# 之前没有文件夹,则抛异常程序停止
if not os.path.exists(trainset_path):
    raise RuntimeError("not have trainset")
if not os.path.exists(valset_path):
    raise RuntimeError("not have valset")

if __name__ == '__main__':
    sys.stdout = Logger("log/train.txt")
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(describe)
    device=torch.device("cuda:2")
    torch.manual_seed(args.seed) #在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    np.random.seed(args.seed)

    # create dataloader
    dataset = MSUDenseLeavesDataset(trainset_path, args.predictions_number)#尝试旋转
    dataloader = DataLoader(dataset, batch_size=24)
    # print(args.dataset_filepath,args.dataset_filepath[:-1])
    # print(args.dataset_filepath,args.dataset_filepath[:-1] + '_eval', args.predictions_number)
    # print(args.dataset_filepath[:-1]+ '_eval')
    # filepath[-:1]莫名奇妙
    # eval_dataloader = DataLoader(MSUDenseLeavesDataset(args.dataset_filepath[:-1] + '_eval\\', args.predictions_number),
    #                              shuffle=True, batch_size=24)
    eval_dataloader = DataLoader(MSUDenseLeavesDataset(valset_path, args.predictions_number),
                                 shuffle=True, batch_size=24)
    # todo totally arbitrary weights
    # model = PyramidNet(n_layers=5, loss_weights=[torch.tensor([1.0])]*5)#, torch.tensor([1.1]), torch.tensor([1.8]),
    model = UNet(n_channels=3, n_classes=1, bilinear=False)  # , torch.tensor([1.1]), torch.tensor([1.8]),
    # torch.tensor([3.2]), torch.tensor([9.0])])
    if args.load_model: #如果加载之前生成的模型参数
        model.load_state_dict(torch.load(args.load_model))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)
    criterion = torch.nn.BCEWithLogitsLoss()
    viz = args.viz_results
    writer_loss_train = SummaryWriter(f'runs/loss_train/{TIMESTAMP}{saveModelName}')
    writer_loss_val = SummaryWriter(f'runs/loss_val/{TIMESTAMP}{saveModelName}')
    # writer_acc_train = SummaryWriter(f'runs/acc_train/{TIMESTAMP}{saveModelName}')
    # writer_acc_val = SummaryWriter(f'runs/acc_val/{TIMESTAMP}{saveModelName}')
    for epoch in range(0, args.epochs):
        count = 0
        sum_acc = 0
        # samples made of image-targets-masks
        for batch_no, (input_batch, targets, masks) in enumerate(dataloader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            # print("Input shape:", input_batch.shape)
            predictions = model(input_batch)

            if batch_no % 10 == 0:  # predictions[-1]是分辨率最高的那个
                print('\n',predictions[-1].max().item(), predictions[-1].min().item(), predictions[-1].sum().item())
                print('\n',torch.sigmoid(predictions[-1]).max().item(), torch.sigmoid(predictions[-1]).min().item(),
                      torch.sigmoid(predictions[-1]).sum().item())
            # print(targets[0].max().item(), targets[0].min().item(), targets[0].sum().item())
            # print(masks[0].max().item(), masks[0].min().item(), masks[0].sum().item())
            # for i in range(len(predictions)):
            #     print(predictions[i].shape, targets[i].shape, masks[i].shape)
            loss = compute_loss(prediction=predictions, target=targets[-1],mask=masks[-1])
            loss.backward()
            # print("Current Loss:", loss.item())
            optimizer.step()  # 执行单步优化

            if batch_no % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_no*24, len(dataset),
                           100. * batch_no * 24 / len(dataset), loss.item()))

                evaluate(model, eval_dataloader)
            torch.save(model.state_dict(), args.save_path+saveModelName+'.pt')
            # visualize result
            if viz:
                with torch.no_grad():
                    predictions = model(input_batch)
                    p = predictions[-1][10, :, :, :]
                    # p = (torch.nn.functional.sigmoid(p) > .5).float()
                    # avoid using sigmoid, it's the same thing
                    print(p.shape, p.max().item(), p.min().item(), p.sum().item())
                    p = (p > 0.).float()
                    p = p.squeeze().cpu().numpy().astype(np.float32)
                    print(p.shape, np.amax(p), np.sum(p), np.amin(p))

                    plt.imshow(p, cmap='Greys')
                    plt.show()
            for p, t, m in zip(predictions, targets[-1], masks[-1]):  # predictions是模型的输出，值是0.265这种，targets和 masks是标签和掩码，值是0或1

                p = (p > 0.).float()
                pixel_acc = (p * m) * t  # 先用掩码做与操作，因为只关注标注了的部分，只有这部分才有label，然后和targets做与，看看有没有标全
                acc = pixel_acc.sum() / t.sum()
                print(
                    f"Accuracy at scale in trainset({p.shape[1]}x{p.shape[2]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")
                count = count + 1
                sum_acc = sum_acc + acc
        # writer_acc_train.add_scalar('acc',
        #                             sum_acc / count,
        #                             global_step=epoch)
        writer_loss_train.add_scalar('loss',
                          loss.item(),
                          global_step=epoch)
        # evalutation
    writer_loss_train.close()
