import os
import sys
import testPic
# import torch.nn as nn
import time
import torch
from logger import Logger
# from utils import device
# from utils import parse_args
from msu_leaves_dataset import MSUDenseLeavesDataset
from torch.utils.data import DataLoader
from pyramid_network import PyramidNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import ruihua

# 目前未使用
def evaluate_experiment(net, device,eval_dataset,writer_acc_val,writer_loss_val,prefix_of_modelName,epoch):
    net.eval()
    count = 0
    sum_recall = 0
    with torch.no_grad():
        for batch_no, (image, targets, masks) in tqdm(enumerate(eval_dataset)):
            image = image.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            predictions = net(image)
            loss = net.compute_multiscale_loss_experiment(predictions, targets, masks)
            print('Eval Loss:', loss.item())

            for p, t, m in zip(predictions, targets, masks):# predictions是模型的输出，值是0.255这种，targets和 masks是标签和掩码，值是0或1
                p = (p>0.).float()
                pixel_recall = (p * m) * t #先用掩码做与操作，因为只关注标注了的部分，只有这部分才有label，然后和targets做与，看看有没有标全
                recall = pixel_recall.sum() / t.sum()
                # print(f"Accuracy at scale in valset({p.shape[2]}x{p.shape[3]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")
                count += 1
                sum_recall += recall

    writer_acc_val.add_scalar(f'recall/{prefix_of_modelName}',
               sum_recall / count,
               global_step=epoch)
    writer_loss_val.add_scalar(f'loss/{prefix_of_modelName}',
                      loss.item(),
                      global_step=epoch)
    final_val_loss=loss.item()
    final_val_recall=sum_recall / count
    return final_val_loss,final_val_recall
    print('*'*50)
    # net.train() # 没用吧



    # train_prefix = "public_and_own_train"  # 训练集所在文件夹名称的前缀 训练集=前缀+patch_size
    # val_prefix = "public_and_own_val"
    # train_prefix = "own_train_aug"  # 训练集所在文件夹名称的前缀 训练集=前缀+patch_size
    # val_prefix = "own_val_aug"
def train_util(test_version,describe,patch_size,dataset_filepath,train_prefix,val_prefix,
               predictions_number,batch_size,which_cuda,seed,name_load_model,time_now,save_path,
               epochs,project_path ,log_interval=10,
               viz_results=False,n_layers=5):



    prefix_of_modelName = f"{time_now}Size{patch_size}{test_version}"  # 保存模型名称的前缀
    name_bestResult_model = f"{time_now}Size{patch_size}Best{test_version}"
    path_save_bestResult_model = save_path + name_bestResult_model + ".pt"
    path_save_lastResult_model=save_path + prefix_of_modelName+ ".pt"# 保存最终模型路径
    path_load_model = save_path + name_load_model + ".pt"
    # 最终使用的训练集和验证集路径
    trainset_path = dataset_filepath + f'{train_prefix}{patch_size}'
    valset_path = dataset_filepath + f'{val_prefix}{patch_size}'

    print(f"start_train:{time.localtime()}")
    print(describe)
    sys.stdout = Logger("log/train.txt")
    # 生成曲线
    writer_loss_train = SummaryWriter(f'runs/loss_train/{prefix_of_modelName}')
    writer_loss_val = SummaryWriter(f'runs/loss_val/{prefix_of_modelName}')
    writer_acc_train = SummaryWriter(f'runs/recall_train/{prefix_of_modelName}')
    writer_acc_val = SummaryWriter(f'runs/recall_val/{prefix_of_modelName}')
    # 之前没有文件夹,则抛异常程序停止
    if not os.path.exists(trainset_path):
        raise RuntimeError("not have trainset")
    if not os.path.exists(valset_path):
        raise RuntimeError("not have valset")

    # create dataloader
    dataset_train = MSUDenseLeavesDataset(trainset_path, predictions_number,random_augmentation=False)
    dataloader = DataLoader(dataset_train, batch_size=batch_size)
    dataset_val=MSUDenseLeavesDataset(valset_path, predictions_number,random_augmentation=False)
    eval_dataloader = DataLoader(dataset_val,batch_size=batch_size)

    # todo totally arbitrary weights
    loss_weights = [torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]),
                    torch.tensor([1.0]), torch.tensor([1.0])]
    model = PyramidNet(n_layers=n_layers, loss_weights=loss_weights)#
    # model = PyramidNet(n_layers=7,
    #                    loss_weights=[torch.tensor([1.0])]*7)#, torch.tensor([1.1]), torch.tensor([1.8]),

    device=torch.device(which_cuda)
    torch.manual_seed(seed) #在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    np.random.seed(seed)
    if path_load_model!=None: #如果加载之前生成的模型参数
        model.load_state_dict(torch.load(path_load_model))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    best_epoch=0
    # 最终指标记录
    final_train_loss=0.
    final_train_recall=0.
    final_val_loss=0.
    final_val_recall=0.

    for epoch in range(0, epochs):
        count = 0
        sum_recall=0
        # samples made of image-targets-masks
        for batch_no, (input_batch, targets, masks) in enumerate(dataloader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            predictions = model(input_batch)

            if batch_no % 10 == 0:  # predictions[-1]是分辨率最高的那个
                print("predictions:max,min,sum",predictions[-1].max().item(), predictions[-1].min().item(), predictions[-1].sum().item())
                print("sigmoid(predictions):max,min,sum",torch.sigmoid(predictions[-1]).max().item(), torch.sigmoid(predictions[-1]).min().item(),
                      torch.sigmoid(predictions[-1]).sum().item(),'\n')
            # for i in range(len(predictions)):
            #     print(predictions[i].shape, targets[i].shape, masks[i].shape)
            loss = model.compute_multiscale_loss_experiment(predictions, targets, masks)
            loss.backward()
            # print("Current Loss:", loss.item())
            optimizer.step()  # 执行单步优化

            if batch_no % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_no*(batch_size), len(dataset_train),
                           100. * batch_no * (batch_size) / len(dataset_train), loss.item()))
                # evaluate(model, eval_dataloader) # 2021.05.05不应该放到这里吧，我放到了后面
            # 更新保存最优数据
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch=epoch
                torch.save(model.state_dict(), path_save_bestResult_model)
            # 100轮保存一次
            if epoch>0 and epoch%100==99:
                saveModelName = prefix_of_modelName + f"Epoch{epoch}.pt"
                path_save_model = save_path + saveModelName
                torch.save(model.state_dict(), path_save_model)
            # visualize result
            if viz_results:
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
                p = (p>0.).float()
                pixel_recall = (p * m) * t #先用掩码做与操作，因为只关注标注了的部分，只有这部分才有label，然后和targets做与，看看有没有标全
                recall = pixel_recall.sum() / t.sum()

                count = count + 1
                sum_recall=sum_recall+recall
                # 注释掉多余的输出，方便查看日志
                # print(f"Accuracy at scale in trainset({p.shape[2]}x{p.shape[3]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")
        # 保存最终的模型
        torch.save(model.state_dict(), f'{save_path+prefix_of_modelName}.pt')
        # tensorboard显示曲线
        writer_acc_train.add_scalar(f'recall/{prefix_of_modelName}',
                                    sum_recall / count,
                                    global_step=epoch)
        writer_loss_train.add_scalar(f'loss/{prefix_of_modelName}',
                          loss.item(),
                          global_step=epoch)
        final_train_loss = loss.item()
        final_train_recall = sum_recall / count
        final_val_loss,final_val_recall=evaluate_experiment(model, device,eval_dataloader,writer_acc_val,writer_loss_val,prefix_of_modelName,epoch)
    # 输出模型名和描述的对应关系
    with open(os.path.join(project_path,"model_des.log"), 'a+') as f:
        f.write(prefix_of_modelName+ "--->" +describe + '\n')
        f.write("final_train_loss:"+str(final_train_loss)+
                ";final_val_loss:"+str(final_val_loss)+
                ";final_train_recall:"+str(final_train_recall)+
                ";final_val_recall:"+str(final_val_recall)+ '\n')
    return model,path_save_bestResult_model,path_save_lastResult_model # 返回模型，最好参数路径，最后参数路径



if __name__ == '__main__':
    project_path = "/home/wangjk/project/pyramid/"
    # dataset_filepath = "/home/wangjk/project/pyramid/data/divide/"
    dataset_filepath = "/home/wangjk/project/pyramid/data/gen/"
    save_path = "/home/wangjk/project/pyramid/modelParameter/"
    time_now=time.strftime("%m-%d_%H-%M", time.localtime())
    log_interval = 10  # 更新记录的频率
    viz_results = False
    seed = 7
    path_load_model = None
    name_load_model = "Epoch1000Pathsize256allownpic"  # 不是从头训练时，加载的模型参数
    is_ruihua = False
    ###################################################################
    # 数据
    train_prefix = "own_train"
    val_prefix = "own_val"
    patch_size = 224
    # 训练
    epochs = 10
    batch_size = 10
    which_cuda = "cuda:3"
    # 测试
    describe = "data:自己数据\t "\
               "train_policy:共100代\t " \
               "network:5层网络,权值均为1.0, 32通道。\n " \
               "aim:测试接口抽象是否正确"
    test_version="test01"
    predictions_number = 5  # 更改网络层数时要搞他
    n_layers=5

    # device = torch.device(which_cuda)


# 训练
    model,path_bestRes,path_lastRes=\
        train_util(test_version=test_version,describe=describe ,patch_size=patch_size,dataset_filepath=dataset_filepath,train_prefix=train_prefix,val_prefix=val_prefix,
                   predictions_number=predictions_number,batch_size=batch_size,which_cuda=which_cuda,seed=seed,name_load_model=name_load_model,time_now=time_now,save_path=save_path,
                   epochs=epochs,project_path = project_path,log_interval=log_interval,
                   viz_results=viz_results,n_layers=n_layers)