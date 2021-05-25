import os
import sys
import testPic
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
import args
import ruihua
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now()) # 添加时间戳解决，曲线显示紊乱的问题
def evaluate(net, eval_dataset):
    net.eval()
    count = 0
    sum_acc = 0
    with torch.no_grad():
        for batch_no, (image, targets, masks) in tqdm(enumerate(eval_dataset)):
            image = image.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            # targets = [targets[0],targets[0],targets[0],targets[0],targets[0]]
            # masks = [ masks[0],masks[0],masks[0],masks[0],masks[0]]
            predictions = net(image)
            loss = net.compute_multiscale_loss(predictions, targets, masks)
            # for i in range(len(predictions)):
            #  print(predictions[i].shape, targets[i].shape, masks[i].shape)
            print('Eval Loss:', loss.item())
            # pixel-wise accuracy of multiscale predictions (edges-only)

            for p, t, m in zip(predictions, targets, masks):# predictions是模型的输出，值是0.255这种，targets和 masks是标签和掩码，值是0或1

                p = (p>0.).float()
                pixel_acc = (p * m) * t #先用掩码做与操作，因为只关注标注了的部分，只有这部分才有label，然后和targets做与，看看有没有标全
                acc = pixel_acc.sum() / t.sum()
                # 注释掉多余的日志，方便定位问题
                # print(f"Accuracy at scale in valset({p.shape[2]}x{p.shape[3]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")
                count = count + 1
                sum_acc=sum_acc+acc
            # p = predictions[-1]
            # t = targets[-1]
            # m = masks[-1]
            # pixel_acc = (p * m) * t
            # acc = pixel_acc.sum() / t.sum()
            # count = count + 1
            # sum_acc = sum_acc + acc
    writer_acc_val.add_scalar('acc',
      sum_acc/count,
      global_step=epoch)
    writer_loss_val.add_scalar('loss',
                      loss.item(),
                      global_step=epoch)
    print('*'*50)
    # net.train() # 没用吧

def evaluate_experiment(net, eval_dataset):
    net.eval()
    count = 0
    sum_recall = 0
    with torch.no_grad():
        for batch_no, (image, targets, masks) in tqdm(enumerate(eval_dataset)):
            image = image.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            predictions = net(image)
            loss = net.compute_multiscale_loss(predictions, targets, masks)
            print('Eval Loss:', loss.item())

            for p, t, m in zip(predictions, targets, masks):# predictions是模型的输出，值是0.255这种，targets和 masks是标签和掩码，值是0或1
                p = (p>0.).float()
                pixel_recall = (p * m) * t #先用掩码做与操作，因为只关注标注了的部分，只有这部分才有label，然后和targets做与，看看有没有标全
                recall = pixel_recall.sum() / t.sum()
                # print(f"Accuracy at scale in valset({p.shape[2]}x{p.shape[3]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")
                count = count + 1
                sum_recall=sum_recall+recall

    writer_acc_val.add_scalar('acc',
      sum_recall/count,
      global_step=epoch)
    writer_loss_val.add_scalar('loss',
                      loss.item(),
                      global_step=epoch)

    print('*'*50)
    # net.train() # 没用吧

if __name__ == '__main__':
    print(f"start_train:{time.localtime()}")
    print(args.describe)
    sys.stdout = Logger("log/train.txt")
    # 最终使用的训练集和验证集路径
    trainset_path = args.dataset_filepath + f'{args.train_prefix}{args.patch_size}'
    valset_path = args.dataset_filepath + f'{args.val_prefix}{args.patch_size}'
    # 之前没有文件夹,则抛异常程序停止
    if not os.path.exists(trainset_path):
        raise RuntimeError("not have trainset")
    if not os.path.exists(valset_path):
        raise RuntimeError("not have valset")

    # create dataloader
    dataset_train = MSUDenseLeavesDataset(trainset_path, args.predictions_number,random_augmentation=True)
    dataloader = DataLoader(dataset_train, batch_size=args.batch_size)
    dataset_val=MSUDenseLeavesDataset(valset_path, args.predictions_number,random_augmentation=False)
    eval_dataloader = DataLoader(dataset_val,batch_size=args.batch_size)

    # todo totally arbitrary weights
    model = PyramidNet(n_layers=5, loss_weights=[torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]),
                                                 torch.tensor([1.0]), torch.tensor([1.0])])#
    # model = PyramidNet(n_layers=7,
    #                    loss_weights=[torch.tensor([1.0])]*7)#, torch.tensor([1.1]), torch.tensor([1.8]),
    device=torch.device(args.which_cuda)
    torch.manual_seed(args.seed) #在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    np.random.seed(args.seed)
    if args.load_model: #如果加载之前生成的模型参数
        model.load_state_dict(torch.load(args.path_load_model))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)

    viz = args.viz_results
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    best_epoch=0
    # 生成曲线
    writer_loss_train = SummaryWriter(f'runs/loss_train/{args.prefix_of_modelName}')
    writer_loss_val = SummaryWriter(f'runs/loss_val/{args.prefix_of_modelName}')
    writer_acc_train = SummaryWriter(f'runs/acc_train/{args.prefix_of_modelName}')
    writer_acc_val = SummaryWriter(f'runs/acc_val/{args.prefix_of_modelName}')
    for epoch in range(0, args.epochs):
        count = 0
        sum_acc = 0
        # pixel_acc_sum=0
        # pixel_target_sum=0
        # samples made of image-targets-masks
        for batch_no, (input_batch, targets, masks) in enumerate(dataloader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            # targets = [targets[0],targets[0],targets[0],targets[0],targets[0]]
            # masks = [masks[0],masks[0],masks[0],masks[0],masks[0]]
            # print("Input shape:", input_batch.shape)
            predictions = model(input_batch)

            if batch_no % 10 == 0:  # predictions[-1]是分辨率最高的那个
                print("predictions:max,min,sum",predictions[-1].max().item(), predictions[-1].min().item(), predictions[-1].sum().item())
                print("sigmoid(predictions):max,min,sum",torch.sigmoid(predictions[-1]).max().item(), torch.sigmoid(predictions[-1]).min().item(),
                      torch.sigmoid(predictions[-1]).sum().item(),'\n')
            # print(targets[0].max().item(), targets[0].min().item(), targets[0].sum().item())
            # print(masks[0].max().item(), masks[0].min().item(), masks[0].sum().item())
            # for i in range(len(predictions)):
            #     print(predictions[i].shape, targets[i].shape, masks[i].shape)
            loss = model.compute_multiscale_loss(predictions, targets, masks)
            loss.backward()
            # print("Current Loss:", loss.item())
            optimizer.step()  # 执行单步优化

            if batch_no % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_no*(args.batch_size), len(dataset_train),
                           100. * batch_no * (args.batch_size) / len(dataset_train), loss.item()))
                # evaluate(model, eval_dataloader) # 2021.05.05不应该放到这里吧，我放到了后面
            # 更新保存最优数据
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch=epoch
                torch.save(model.state_dict(), args.path_save_bestResult_model)
            # 100轮保存一次
            if epoch>0 and epoch%100==99:
                saveModelName = args.prefix_of_modelName + f"Epoch{epoch}.pt"
                path_save_model = args.save_path + saveModelName
                torch.save(model.state_dict(), path_save_model)
            # visualize result
            if viz:
                with torch.no_grad():
                    predictions = model(input_batch) # 应该可以注释掉，上面已经出现过一次了
                    p = predictions[-1][10, :, :, :] # 只显示最高分辨率的一个
                    # p = (torch.nn.functional.sigmoid(p) > .5).float()
                    # avoid using sigmoid, it's the same thing
                    print(p.shape, p.max().item(), p.min().item(), p.sum().item())
                    p = (p > 0.).float()
                    p = p.squeeze().cpu().numpy().astype(np.float32)
                    print(p.shape, np.amax(p), np.sum(p), np.amin(p))

                    plt.imshow(p, cmap='Greys')
                    plt.show()
            for p, t, m in zip(predictions, targets, masks):  # predictions是模型的输出，值是0.265这种，targets和 masks是标签和掩码，值是0或1

                p = (p > 0.).float()
                pixel_acc = (p * m) * t  # 先用掩码做与操作，因为只关注标注了的部分，只有这部分才有label，然后和targets做与，看看有没有标全
                acc = pixel_acc.sum() / t.sum()
                # 注释掉多余的输出，方便查看日志
                # print(
                #     f"Accuracy at scale in trainset({p.shape[2]}x{p.shape[3]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")
                count = count + 1
                sum_acc = sum_acc + acc
            # p=predictions[-1]
            # t=targets[-1]
            # m=masks[-1]
            # pixel_acc = (p * m) * t # 逐像素乘法
            # acc=pixel_acc.sum() / t.sum()
            # count = count + 1
            # sum_acc = sum_acc + acc

        writer_acc_train.add_scalar(f'recall/{args.prefix_of_modelName}',
                                    sum_acc / count,
                                    global_step=epoch)
        writer_loss_train.add_scalar(f'loss/{args.prefix_of_modelName}',
                          loss.item(),
                          global_step=epoch)
        evaluate(model, eval_dataloader)
    # 输出模型名和描述的对应关系
    with open(os.path.join(args.project_path,"model_des.log"), 'a+') as f:
        f.write(args.prefix_of_modelName+ "--->" +args.describe + '\n')
    writer_loss_train.close()
    # 测试
    testPic.test_picture(args.dirpath_orginpics,args.bin_outputPath,model,args.path_save_bestResult_model,args.name_bestResult_model)
