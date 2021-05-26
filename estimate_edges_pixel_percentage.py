import cv2
import numpy as np
import glob

# 计算数据集中所有图像的边缘像素和内部像素的比例，用于给边缘和内部像素赋予不同的权值
if __name__=="__main__":
    labels = sorted(glob.glob('/home/wangjk/project/pyramid/data/gen/train/*_label.png'))
    masks = sorted(glob.glob('/home/wangjk/project/pyramid/data/gen/train/*_mask.png'))
    # print(len(labels),len(masks)) 892 892

    total = 0
    for i, (l, m) in enumerate(zip(labels, masks)):
        li = cv2.imread(l)[:, :, 0]
        mi = cv2.imread(m)[:, :, 0]
        # print(li.shape, mi.shape)
        # count number of edges
        n_edges = np.count_nonzero(li)  # 边像素
        n_pixels = li.shape[0] * li.shape[1] # 总体像素数
        n_not_edges = np.count_nonzero(mi) - n_edges
        assert n_not_edges > n_edges
        # print("Edges over interior pixels:", (n_edges/n_not_edges))
        total += (n_edges/n_not_edges)
        print(i)

    print("Edges over interior pixels (mean):", (total/len(labels)))