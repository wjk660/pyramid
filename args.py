
#train: 注意更改：epochs和saveModelName的值,train_prefix,val_prefix
import time

describe="data:只使用自己数据，224*224，分离train和val的数据集，不设置随机翻转。" \
         "train_policy:从头开始训练，共10代，100代保存一次，最优结果为Best,无锐化，batch_size=24。" \
         "network:5层网络,权值均为1.0。" \
         "aim:测试改tensorboard"
         # "aim:测试改loss公式后测试图像为纯黑，改回loss看看哪里的问题"
# "aim:label不进行缩放,改动的dataset.py"
# ”测试改acc,测试改变channel的影响“
# "aim:对比测试中间层loss计算的重要性，随机翻转0.2。"只有最底层权值为1.0，其它均为0.1
# "network:Unet,bilinear=True"\

# train_prefix = "public_and_own_train"  # 训练集所在文件夹名称的前缀 训练集=前缀+patch_size
# val_prefix = "public_and_own_val"
train_prefix = "own_train"  # 训练集所在文件夹名称的前缀 训练集=前缀+patch_size
val_prefix = "own_val"
patch_size = 224
is_ruihua=False

load_model=False
epochs =10
batch_size=24
which_cuda="cuda:1"

log_interval = 10  # 更新记录的频率
predictions_number = 5 # 更改网络层数时要搞他

dataset_filepath = "/home/wangjk/project/pyramid/data/gen/"
seed = 7
save_path = "/home/wangjk/project/pyramid/modelParameter/"
viz_results = False

time_now = time.strftime("%m-%d_%H-%M", time.localtime())
project_path = "/home/wangjk/project/pyramid/"

prefix_of_modelName = f"{time_now}Size{patch_size}" #保存模型名称的前缀
name_bestResult_model=f"{time_now}Size{patch_size}Best"
path_save_bestResult_model=save_path+name_bestResult_model+".pt"
name_load_model="Epoch1000Pathsize256allownpic"# 不是从头训练时，加载的模型参数
path_load_model=save_path+name_load_model+".pt"
# 测试效果
dirpath_orginpics = "/home/wangjk/project/pyramid/data/output/origin"
bin_outputPath = "/home/wangjk/project/pyramid/data/output/bin"
