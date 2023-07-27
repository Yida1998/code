"""
Author: yida
Time is: 2022/8/18 10:41 
this Code: 分析预测结果: 子图标签, 子图结果, 原图标签, 原图结果
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    path = r"D:\Desktop\0821\TrainSet2_224_dropm0.0_adaption_1.0\2058\test_dict.npy"
    data = np.load(path, allow_pickle=True)
    sub_true = data[0]
    sub_pre = data[1]
    original_true = data[2]
    original_pre = data[3]
    # 原图混淆矩阵

    y_true = original_true
    y_fit = original_pre

    # 1.统计报告:precision recall f1-score accuracy support
    # ================================================================= #
    #       y_true : 真实标签  y_fit = 预测结果 y_prob:预测结果的概率
    # ================================================================= #
    #       target_name : 类标签
    print("#=============================1.support===============================#")
    target_names = ['暗紫泥二泥土', '暗紫泥大泥土', '暗紫泥夹砂土', '暗紫泥油石骨子土', '灰棕紫泥半砂半泥土', '灰棕紫泥大眼泥土', '灰棕紫泥石骨子土', '灰棕紫泥砂土', '红棕紫泥红棕紫泥土', '红棕紫泥红石骨子土', '红棕紫泥红紫砂泥土']

    support = classification_report(y_true, y_fit, target_names=target_names)
    print(support)
    #  support是str类型,不方便取值,如果想要取值的话可以 转换成字典类型:output_dict = True
    support_dict = classification_report(y_true, y_fit, target_names=target_names, output_dict=True)
    # 转换成字典取值
    for k, v in support_dict.items():
        print(k, v)
    print("accuracy:", support_dict['accuracy'])
    # 单独求准确率指标
    accuracy = accuracy_score(y_true, y_fit)
    print("accuracy:", accuracy)
    # 2.计算混淆矩阵: confusion_matrix
    # ================================================================= #
    #       参考链接:https://blog.csdn.net/qq_36264495/article/details/88074246
    #       样式设计:https://blog.csdn.net/ztf312/article/details/102474190
    # ================================================================= #
    print("#==========================2.计算混淆矩阵=======================================#")
    matrix = confusion_matrix(y_true, y_fit)
    plt.matshow(matrix, cmap='YlOrRd')
    plt.title("Confusion_Matrix")
    plt.show()
    print("混淆矩阵:\n", matrix)
