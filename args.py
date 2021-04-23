dataset_filepath = "/home/wangjk/dataset/DenseLeaves/gen/"
epochs = 500
seed = 7
log_interval = 10
predictions_number = 5
save_path = "/home/wangjk/project/pyramid/modelParameter/"
# load_model = "pyramid_net_model.pt"
load_model = False
viz_results = False
patch_size = 256
train_prefix = "own_instance_all"  # 训练集所在文件夹名称的前缀 训练集=前缀+patch_size
val_prefix = "own_instance_val"