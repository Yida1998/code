"""
Author: yida
Time is: 2022/4/22 19:57 
this Code: 计数土壤图像
"""
import os

import cv2

path = "/Users/yida/Desktop/实验数据/实验三/train/600/val"
file = os.listdir(path)
if '.DS_Store' in file:
    file.remove('.DS_Store')
sum = 0
for i in file:
    if '.' not in i:
        sub_path = os.path.join(path, i)
        file_sub = os.listdir(sub_path)
        if '.DS_Store' in file_sub:
            file_sub.remove('.DS_Store')
    # 判断大小
    # for j in file_sub:
    #     sub_path_j = os.path.join(sub_path, j)
    #     img = cv2.imread(sub_path_j, 0)
    #     # print(img.shape)
    #     if img.shape != (3000, 4000):
    #         print(sub_path_j)
        print("{}: => {}".format(i, len(file_sub)))
        sum += len(file_sub)
print("一共采集到{}张土壤数字图像".format(sum))