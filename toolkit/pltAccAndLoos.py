"""
Author: yida
Time is: 2021/12/18 09:49 
this Code: 读取npy中的数据, 绘制损失以及准确率识别曲线
"""
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    path = '/Users/yida/Desktop/code/Tools/Variable/0523/TrainSet2_r004/1128'
    loss_path = path + '/1var_loss.npy'
    acc_path = path + '/1var_acc.npy'
    epoch = 120

    plt.figure()
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    loss = np.load(loss_path)
    step = len(loss) // epoch  # s
    loss_step = loss[::step]  # 每个epoch中的一个step
    loss_epoch = np.zeros(epoch)  # 每个epoch的损失之和

    for i in range(epoch):
        loss_epoch[i] = np.sum(loss[i * step:step * (i + 1)])
    plt.plot(loss_epoch, 'g')
    plt.title("Loss Curve")
    plt.ylabel('Loss')
    plt.xlabel('Step')

    # 绘制acc曲线
    plt.subplot(1, 2, 2)
    acc = np.load(acc_path)
    plt.plot(acc, 'r')
    plt.title("ACC Curve")
    plt.ylabel('Acc')
    plt.xlabel('Step')
    plt.show()
