"""
Author: yida
Time is: 2022/8/16 16:06 
this Code: 读取npy中的数据, 绘制损失以及准确率识别曲线; 多条曲线同时进行; 分别传入损失曲线,准确率曲线所保存的npy路径
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置字体为宋体
config = {
    "font.family": 'serif',
    # "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['Songti SC'],
}
rcParams.update(config)


def pltLossFigure(loss_path):
    """
    绘制损失曲线
    :param loss_path:
    :return:
    """
    plt_num = len(loss_path)  # 一共需要绘制多少个图

    plt.subplot(1, 2, 1)
    for i in range(plt_num):
        loss = np.load(loss_path[i])
        step = len(loss) // epoch  # s
        loss_step = loss[::step]  # 每个epoch中的一个step
        loss_epoch = np.zeros(epoch)  # 每个epoch的损失之和
        for j in range(epoch):
            loss_epoch[j] = np.sum(loss[j * step:step * (j + 1)])
        plt.plot(loss_epoch, label=label_item[i])
    # plt.title("Loss Curve")
    plt.ylabel('损失值')
    plt.xlabel('训练次数')
    plt.yticks(fontproperties='Times New Roman')
    plt.xticks(fontproperties='Times New Roman')


def pltAccFigure(acc_path):
    """
    绘制准确率曲线
    :param acc_path:
    :return: None
    """
    acc_num = len(acc_path)
    plt.subplot(1, 2, 2)
    for i in range(acc_num):
        acc = np.load(acc_path[i])
        plt.plot(acc, label=label_item[i])
    # plt.title("Acc Curve")
    plt.ylabel('验证集准确率')
    plt.xlabel('训练次数')
    plt.tight_layout()  # 防止重叠
    plt.legend()  # 显示label等
    plt.yticks(fontproperties='Times New Roman')
    plt.xticks(fontproperties='Times New Roman')
    plt.savefig("./result.svg", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    epoch = 120
    plt.figure()  # 初始化幕布
    label_item = ['α=0.5', 'α=1.0', 'α=1.5', 'α=2.0']
    # 绘制损失曲线
    loss_paths = ['/Users/yida/Desktop/实验结果/实验二/0830/train/0000/1var_loss.npy',
                  '/Users/yida/Desktop/实验结果/实验二/0830/train/0049/1var_loss.npy',
                  '/Users/yida/Desktop/实验结果/实验二/0830/train/0128/1var_loss.npy',
                  '/Users/yida/Desktop/实验结果/实验二/0830/train/0218/1var_loss.npy']

    # 绘制准确率曲线
    acc_paths = ['/Users/yida/Desktop/实验结果/实验二/0830/train/0000/1var_acc.npy',
                 '/Users/yida/Desktop/实验结果/实验二/0830/train/0049/1var_acc.npy',
                 '/Users/yida/Desktop/实验结果/实验二/0830/train/0128/1var_acc.npy',
                 '/Users/yida/Desktop/实验结果/实验二/0830/train/0218/1var_acc.npy']

    pltLossFigure(loss_paths)
    pltAccFigure(acc_paths)
