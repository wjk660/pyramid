import sys
import train_utils
import time
import torch
import testPic
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
        train_utils.train_util(test_version=test_version,describe=describe ,patch_size=patch_size,dataset_filepath=dataset_filepath,train_prefix=train_prefix,val_prefix=val_prefix,
                   predictions_number=predictions_number,batch_size=batch_size,which_cuda=which_cuda,seed=seed,name_load_model=name_load_model,time_now=time_now,save_path=save_path,
                   epochs=epochs,project_path = project_path,log_interval=log_interval,
                   viz_results=viz_results,n_layers=n_layers)

# 测试
    dirpath_orginpics = "/home/wangjk/project/pyramid/data/output/origin"
    bin_outputPath = "/home/wangjk/project/pyramid/data/output/bin"
    testPic.test_picture(dirpath_orginpics,bin_outputPath,model,path_bestRes)
    testPic.test_picture(dirpath_orginpics,bin_outputPath,model,path_lastRes)
    sys.exit(0)