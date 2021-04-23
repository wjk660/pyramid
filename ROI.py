import matplotlib.pyplot as plt
import numpy as np
import cv2
# 功能：利用分割时生成的bin，生成mix图像。然后切分程单个叶片leave，矩形框rectangleImg。\
# 修改内容：功能中提到的各个文件路径
# 图像位置：classcification文件夹下，
# --------------------孔洞填充-------------------------
# https://www.icode9.com/content-1-595450.html
def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out
# 图像最外周围均设为255
def  AddWhiteEdge(img,pic_width,pic_height):
    for l in range(pic_height):
        img[0, l,:] = 255
        img[pic_width - 1, l,:] = 255
    for w in range(pic_width):
        img[w, 0,:] = 255
        img[w, pic_height - 1,:] = 255
#
# ----------------------------得到的label图像和合并----------------------------
import glob
import os
from os.path import isdir
import shutil
origin_path="/home/wangjk/dataset/DenseLeaves/classification/origin"
bin_path="/home/wangjk/dataset/DenseLeaves/classification/bin/epoch200"
mix_path="/home/wangjk/dataset/DenseLeaves/classification/mix/epoch200"
leave_path="/home/wangjk/dataset/DenseLeaves/classification/leave/epoch200"
rectangle_path="/home/wangjk/dataset/DenseLeaves/classification/rectangleImg"
if not os.path.exists(origin_path):
    raise("not folder or no picture")
if not os.path.exists(bin_path):
    os.mkdir(bin_path)
    raise("not folder or no picture")
if not os.path.exists(mix_path):
    os.mkdir(mix_path)
    raise("not folder ")
if not os.path.exists(leave_path):
    os.mkdir(leave_path)
    raise("not folder ")
if not os.path.exists(rectangle_path):
    os.mkdir(rectangle_path)
    raise("not folder ")
# 执行代码前清空指定的文件夹
shutil.rmtree(mix_path)
os.mkdir(mix_path)
shutil.rmtree(leave_path)
os.mkdir(leave_path)

list_originImgsPath = [image_origin for image_origin in glob.glob(origin_path + "/*.JPG")]  # 读出所有文件夹的路径
for i in range(len(list_originImgsPath)):
    img_origin = cv2.imread(list_originImgsPath[i])
    name_img = list_originImgsPath[i].split('/')[-1]
    img_bin=cv2.imread(os.path.join(bin_path, f'bin_{name_img}'))
    pic_width = img_bin.shape[0]
    pic_height = img_bin.shape[1]
    AddWhiteEdge(img_bin,pic_width,pic_height) # todo:应该是给mask加白边
    gray = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
    # # 转为2值图
    ret,mask = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    mask_inv=cv2.bitwise_not(mask)
    img_mix = cv2.bitwise_and(img_origin,img_origin,mask = mask_inv)
    cv2.imwrite(os.path.join(mix_path,f"mix_{name_img}"), img_mix)
    # plt.imshow(gray)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(mask_inv)
    # plt.show()
    # plt.imshow(img_mix)
    # plt.show()
    # ----------------------------产生图像框----------------------------
    # 寻找轮廓
    contours, hier = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count=0
    for cidx, cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        if (w * h > 4000 and not (w == pic_height and h == pic_width)):  # 大于阈值，且不是最外边的框
            print('RECT: x={}, y={}, w={}, h={}'.format(x, y, w, h))
            # 原图绘制圆形
            cv2.rectangle(img_mix, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 25), thickness=3)
            count = count + 1
            # 截取ROI图像
            originMinPic = img_origin[y:y + h, x:x + w, :]
            cv2.imwrite(os.path.join(leave_path, f'{x}_{y}_PicOrigin.JPG'), originMinPic)
            maskBlock = mask[y:y + h, x:x + w]
            maskBlock_inv = mask_inv[y:y + h, x:x + w]
            # plt.imshow(originMinPic)
            # plt.show()
            # plt.imshow(maskBlock)
            # plt.show()
            # plt.imshow(maskBlock_inv)
            # plt.show()
            mask_out = FillHole(maskBlock)  # 孔洞填充
            # plt.imshow(mask_out)
            # plt.show()
            MaskFinal = cv2.bitwise_and(mask_out, maskBlock_inv)
            # plt.imshow(MaskFinal)
            # plt.show()
            cv2.imwrite(os.path.join(leave_path, f'{x}_{y}_MaskFinal.JPG'), MaskFinal)
            PicFinal = cv2.bitwise_and(originMinPic, originMinPic, mask=MaskFinal)
            cv2.imwrite(os.path.join(leave_path, f'{x}_{y}_PicFinal.JPG'), PicFinal)
            # plt.imshow(PicFinal)
            # plt.show()
        # if count>2:
        #     print("够了")
        #     break
    print(count)
    cv2.imwrite(os.path.join(rectangle_path, f"Rectangle_4000_{name_img}"), img_mix)


