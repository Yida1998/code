"""
Author: yida
Time is: 2022/3/20 10:37 
this Code: 单个训练文件夹

2022年04月09日23:56:46
--------------------------最终说明---------------------------
copy trainBatch.py  ->  把它改成针对 单个数据集操作的

说明:train.py是基于trainBatch.py进行修改的, 尽可能与它保存一致; 所以在新增trainBatch.py功能时, 应修改本文件

train_path:训练集路径
val_path: 验证集路径, 当它为空时, 默认与测试集路径相同
test_path: 测试集路径, 最终保存模型的测试路径
-----------------------------------------------------------

2022年01月04日09:55:10
1.自己写的model, BN层有误
2.使用model.eval会使测试更加波动, 用标准的Resnet最终还是会收敛
3.torch.no_grad() 和 eval 谁在前后 无所谓, 用大的作用域 包含小的作用域更好呢
2022年03月08日16:00:18
1.今天来从新回顾下原来的代码, 把不足的地方进行补充和修改 -> 已完成
2.看上一次留言, 你的resnet没有进行凯明初始化导致的  -> 已完成
问题待修改:
1.模型选类别的时候, 直接用名字, 不要用0, 1, 2分不清, 自己统一下叫网络的名称  -> 已完成
2.学习率策略加上, 整一个参数, 可选择的      -> 已完成
3.Normalize应该把它计算出来         -> 已完成, 这个很OK 有数据随机可以算 ./Tools/toolkit
4.每一句输出是不是应该写入到TXT中去    -> 已完成 全部写入到result.txt中去
5.把不要的文件删掉, 然后把服务器上的结果覆盖下来  -> 已完成 已经删除 我在检查一次
6.K-Fold 以及批训练整理一下, 以及找出每个k折里面最好的模型, 输出的结果都要保存到txt中下来
7.整体代码 不合理的改进一下 -> 整体还行吧, 能够正常运行, 没啥太大的错误
8.训练的时候把准确率及损失的图像画出来 -> 已完成, 新写了个函数, 全部封装在里面的

问题待完成
1.损失还是应该统计epoch的, step容易出错, 而且横坐标epoch不能保证相同

2022年08月29日15:12:28
新增输出测试集路径
"""
import argparse
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision.models import alexnet  # alexnet
from torchvision.models import resnet18  # resnet18

from Model.Resnet18 import ResNet18  # resnet
from Model.ResnetSe18 import ResNetSe18  # resnet-se
from Model.efficientnet import efficientnet_b7  # efficientnet_b0
from Model.mobilenetv3 import mobilenet_v3_small  # mobile_netv3 small
from Model.shufflenetv2 import shufflenet_v2_x1_0  # shufflenet_v2
from Model.testModel import TestModel  # testnet
from Model.vit_model import vit_base_patch16_224  # vit-224_16
from toolkit.PredictDataSet import GetMyData  # 测试阶段自定义数据集

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def randomSeed(seed=2022):
    """
    为重复实验设计固定的随机数种子
    :param seed:随机数种子 2021
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def initialization():
    # 输出Gpu信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        print("The device is:", gpu_name)
    else:
        print("The device is:Cpu-Only")
    # 设置随机数种子
    randomSeed()
    print("开始时间:", time.strftime('%Y-%m-%d-%H:%M'))
    # 初始化文件夹
    UID = time.strftime('/%m%d/' + opt.train_path.split('/')[-1] + '/%H%M')  # 唯一标识符
    # modelSave = opt.model_path + UID  # 模型存储路径
    # varSave = opt.var_path + UID  # 变量存储路径
    # 2021年12月15日23:04:41 修改统一工具类保存路径, 仅用一个地址保存
    modelSave = opt.Tools_path + '/Model' + UID
    varSave = opt.Tools_path + '/Variable' + UID
    if os.path.exists(modelSave):
        shutil.rmtree(modelSave)
        os.makedirs(modelSave)
        print("已存在, 删除并成功创建:", modelSave)
    else:
        os.makedirs(modelSave)
        print("成功创建:", modelSave)
    if os.path.exists(varSave):
        shutil.rmtree(varSave)
        os.makedirs(varSave)
        print("已存在, 删除并成功创建:", varSave)
    else:
        os.makedirs(varSave)
        print("成功创建:", varSave)
        #  ====================输出配置信息====================
    print(
        "训练次数epoches:{} \n重复次数trains_times:{} \n训练集路径train_path:{} --batch:{} \n验证集路径val_path:{} --batch{}\n测试集路径test_path:{} --batch:{} \n学习率learning_rate:{} \n数据增强:{}  打乱数据:{} \n多线程num_workers:{} \n模型名称:{} \nFL损失函数:{}(default交叉熵) "
        "\nStep:{} "
        "\n在Epoch开始验证保存模型:{"
        "}\n是否开启混合精度:{}".format(
            opt.epoches, opt.train_times, train_path, opt.batchSize,
            val_path, opt.batchSize, opt.test_path, opt.batchSize, opt.learning_rate, opt.enhance, opt.shuffle, opt.num_workers, opt.model_name,
            opt.FL, opt.step, opt.leastEpoch, opt.mixAcc))
    print("学习率策略:{}".format(opt.lr_scheduler))
    print("weight_decay:{}".format(opt.weight_decay))
    if opt.raw_image:
        print("使用原始图像raw_image:{}".format(opt.raw_image))
        print("等比缩放大小raw_size:{}".format(opt.raw_size))
    print("-----------初始化已完成...正在启动...-----------")
    return modelSave, varSave


def data_enhance():
    """
    是否进行数据增强
    :return:
    """
    if opt.enhance:
        if opt.raw_image:  # 使用原图的话
            data_transform = transforms.Compose([
                transforms.Resize([3000, 4000]),
                transforms.Resize(opt.raw_size),
                transforms.RandomHorizontalFlip(p=0.25),  # 以概率P水平翻转
                transforms.RandomVerticalFlip(p=0.25),  # 以概率P垂直翻转  都是翻转操作  旋转会有黑边,所以展示不考虑
                # transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0),  # 亮度随机增强百分之50
                transforms.ToTensor(),
                transforms.Normalize((0.49, 0.37, 0.31), (0.16, 0.15, 0.14)),
            ])
        else:  # 使用子图
            data_transform = transforms.Compose([
                transforms.Resize([224, 224]),
                # transforms.RandomHorizontalFlip(p=0.25),  # 以概率P水平翻转
                # transforms.RandomVerticalFlip(p=0.25),  # 以概率P垂直翻转  都是翻转操作  旋转会有黑边,所以展示不考虑
                # transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0),  # 亮度随机增强百分之50
                transforms.ToTensor(),
                transforms.Normalize((0.49, 0.37, 0.31), (0.16, 0.15, 0.14)),
            ])

    else:
        if opt.raw_image:  # 使用原图的话
            data_transform = transforms.Compose([
                transforms.Resize([3000, 4000]),
                transforms.Resize(opt.raw_size),
                transforms.ToTensor(),
                transforms.Normalize((0.49, 0.37, 0.31), (0.16, 0.15, 0.14)),
                # 2021年12月27日20:31:21 一定要加上标准化 避免训练和测试数据因分布不同的问题
            ])
        else:  # 使用子图
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.49, 0.37, 0.31), (0.16, 0.15, 0.14)),
                # 2021年12月27日20:31:21 一定要加上标准化 避免训练和测试数据因分布不同的问题
            ])
    return data_transform


def choseModel(model_name):
    """
    依据模型名称的名称进行模型选择
    :param model_name:
    :return:
    """
    if model_name == 'resnet':
        return ResNet18(n_class=opt.n_class)
    elif model_name == 'alexnet':
        # 2021年12月16日10:37:25 新增模型->AlexNet
        return alexnet(pretrained=False, num_classes=opt.n_class)
    elif model_name == 'resnet18':
        return resnet18(pretrained=False, num_classes=opt.n_class)
    elif model_name == 'resnet-se':
        return ResNetSe18(n_class=opt.n_class)  # 2022年01月03日16:00:09 新增ResnetSe模块
    elif model_name == 'testnet':
        return TestModel(n_class=opt.n_class)  # 2022年01月08日19:43:35 新增测试model
    elif model_name == 'vit':
        return vit_base_patch16_224(num_classes=opt.n_class)
    elif model_name == 'mobilenetv3':
        return mobilenet_v3_small(num_classes=opt.n_class)
    elif model_name == 'shufflenetv2':
        return shufflenet_v2_x1_0(num_classes=opt.n_class)
    elif model_name == 'efficientnetb7':
        return efficientnet_b7(num_classes=opt.n_class)
    elif model_name == 'densenet121':
        return torchvision.models.densenet121(num_classes=opt.n_class)
    elif model_name == 'googlenetv1':
        return torchvision.models.googlenet(num_classes=opt.n_class)
    else:
        print("Which Modul are you need...")
        f.write("Which Modul are you need...\n")


def choseLossFunction():
    """
    默认使用交叉熵损失函数, 留个接口扩展使用FL损失函数
    :return:
    """
    if opt.FL:
        print("选择FLoss...代补充")
        f.write("选择FLoss...代补充\n")
        loss_Function = None
    else:
        loss_Function = nn.CrossEntropyLoss()  # 交叉熵损失函数
        loss_Function.to(device)
    return loss_Function


def choselrScheduler(optimizer):
    """
    选择学习率策略, 默认为None, 不使用任何学习率策略
    :return:
    """
    if opt.lr_scheduler == 'step':
        # print("学习率策略为: steoLr...")
        # f.write("学习率策略为: steoLr...\n")
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif opt.lr_scheduler == 'exponential':
        print("学习率策略为: exponentialLR...")
        f.write("学习率策略为: exponentialLR...\n")
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


def main():
    # 数据变换
    transform = data_enhance()
    # 加载数据集
    dataset_train = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    dataset_val = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    # DataLoader
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batchSize, num_workers=opt.num_workers,
                                  shuffle=opt.shuffle, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=opt.batchSize, num_workers=opt.num_workers,
                                shuffle=opt.shuffle, pin_memory=True)
    # 初始化模型
    model = choseModel(opt.model_name)  # 设置模型名称导入对应的model
    model = model.to(device)  # 送到GPU上 与model.to(device)等价
    # 初始化损失函数
    loss_Function = choseLossFunction()
    # 初始化优化器
    optimizter = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    # 选择学习率策略
    choselrScheduler(optimizer=optimizter)
    #   ---------------------------------新增模块:2021年12月12日20:10:52------------------------------------#
    #                                       判断是否使用自动混合精度
    if opt.mixAcc:
        enable_amp = True if "cuda" in device.type else False
        # 在训练最开始之前实例化一个GradScaler对象
        scaler = amp.GradScaler(enabled=enable_amp)
    #   ---------------------------------------------------------------------#
    # 开始训练
    # =========================中间变量Variable存储=========================
    var_loss = []  # 保存loss
    var_acc = []  # 保存准确率曲线
    best_acc = 0  # 最佳acc
    for epoch in range(opt.epoches):
        # 训练
        model.train()
        running_loss = 0  # 保存总loss
        for step, data in enumerate(dataloader_train):
            optimizter.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Gpu加速
            # -------------开启混合精度-------------
            if opt.mixAcc:
                # 前向过程(model + loss)开启 autocast
                with amp.autocast(enabled=enable_amp):
                    outputs = model(inputs)
                    loss = loss_Function(outputs, labels)
                # 1.将梯度放大防止梯度消失
                scaler.scale(loss).backward()
                # 2.将梯度值unscale回来
                scaler.step(optimizter)
                # 3.准备着 看是否要增大scaler
                scaler.update()
                # 4.正常更新权重
                optimizter.zero_grad()
            # -------------开启混合精度-------------
            else:
                outputs = model(inputs)
                loss = loss_Function(outputs, labels)
                loss.backward()
                optimizter.step()
            running_loss += loss.item()  # 总损失
            # 输出信息
            if step % opt.step == opt.step - 1:
                print("Epoch [{}/{}], Iter[{}/{}], loss: {}".format(epoch + 1, opt.epoches, step + 1,
                                                                    len(dataset_train) // opt.batchSize + 1,
                                                                    loss.item()))
                f.write("Epoch [{}/{}], Iter[{}/{}], loss: {}\n".format(epoch + 1, opt.epoches, step + 1,
                                                                        len(dataset_train) // opt.batchSize + 1,
                                                                        loss.item()))
            var_loss.append(loss.item())  # 保存中间变量, 统计的是每个step的epoch
        # 从统计每个step的loss到统计每个epoch的loss
        # var_loss.append(running_loss)

        # 当epoch大于leastEpoch时才开始进行验证:减少训练时间
        if epoch >= int(opt.epoches * opt.leastEpoch):
            # 测试
            with torch.no_grad():  # 2022年01月04日09:53:02 大的作用域在外面 然后接model.eval 其实谁在前后并不太影响
                model.eval()  # 2021年12月21日21:27:41 开始eval 进行测试并保存模型 eval时必须将输入数据标准化 训练测试同分布,不然BN层后波动严重
                var_target = []  # 放标签
                var_predict = []  # 放预测结果
                for step_, data in enumerate(dataloader_val):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predict = torch.argmax(outputs, dim=1)
                    # 把标签和预测结果放入对应列表, 然后使用sklearn进行acc统计
                    var_target += labels.cpu()  # gpu tensor, 要先转换成cpu tensor 然后才可以转换成numpy
                    var_predict += predict.cpu()
                acc = accuracy_score(var_predict, var_target)
                var_acc.append(acc)
                print("the epoch of {} Prediction Accuracy is {}".format(epoch + 1, acc))
                print("")
                f.write("the epoch of {} Prediction Accuracy is {}\n".format(epoch + 1, acc))
                f.write("\n")
                if acc >= best_acc:
                    # 最佳准确率
                    best_acc = acc
                    bestacc_epoch = epoch
                    # 保存最佳模型
                    torch.save(model.state_dict(), modelSvae + '/best' + str(k) + '.pth')
    # 保存最后训练的模型及参数
    torch.save(model.state_dict(), modelSvae + '/last' + str(k) + '.pth')
    np.save(modelSvae + '/' + str(k + 1) + 'TimesBestEpIs' + str(bestacc_epoch) + '.npy', bestacc_epoch)  # 保存最佳模型名称
    np.save(varSave + '/' + str(k + 1) + 'var_acc.npy', np.array(var_acc))
    np.save(varSave + '/' + str(k + 1) + 'var_loss.npy', np.array(var_loss))
    # 绘制准确率曲线
    draw_loss_acc([varSave, k + 1], var_loss, var_acc)

    print(
        "RunningLoss is :{:.4f} - BestAcc is :{:.4f} - Last_Acc is :{:.4f}".format(running_loss / (step + 1), best_acc,
                                                                                   acc))
    f.write("RunningLoss is :{:.4f} - BestAcc is :{:.4f} - Last_Acc is :{:.4f}\n".format(running_loss / (step + 1),
                                                                                         best_acc,
                                                                                         acc))
    acc_global.append(acc.item())


def Summarize():
    """
    总结信息, 输出times次试验的评价准确率
    :return:
    """
    acc_summarize = np.array(acc_global)
    ModelNum_list.append(acc_summarize)
    print("-------------------------------------------------------------------")
    print("------|Average accuracy of {} the Finally is: {:.4f}±{:.4f}|------".format(opt.train_times,
                                                                                      np.mean(acc_summarize),
                                                                                      np.std(acc_summarize)))
    f.write("-------------------------------------------------------------------\n")
    f.write("------|Average accuracy of {} the Finally is: {:.4f}±{:.4f}|------\n".format(opt.train_times,
                                                                                          np.mean(acc_summarize),
                                                                                          np.std(acc_summarize)))
    avg_std = "{:.4f}±{:.4f}".format(np.mean(acc_summarize), np.std(acc_summarize))  # 最终的结果
    # np.savetxt(varSave + '/result.txt', acc_summarize, fmt='%.4f')  # 保存结果到txt
    # with open(varSave + '/result.txt', 'a') as f2:  # 追加最终结果
    np.savetxt(f, acc_summarize, fmt='%.4f')
    f.write(avg_std + "\n")

    return acc_summarize  # 返回times次训练结果的均值 | 单个训练返回的直接是整个文件夹


def temp_write():
    """
    因你的程序代码 未满足高内聚低耦合原则 较难扩展 因此专门写个函数来存放输出到txt中
    :return:
    """
    f.write("模型存放:{}\n".format(modelSvae))
    f.write("变量存放:{}\n".format(varSave))
    #  ====================输出配置信息====================
    f.write(
        "训练次数epoches:{} \n重复次数trains_times:{} \n训练集路径train_path:{} --batch:{} \n验证集路径val_path:{} --batch:{}\n测试集路径test_path:{} --batch:{} \n学习率learning_rate:{} \n数据增强:{}  打乱数据:{} \n多线程num_workers:{} \n模型名称:{} \nFL损失函数:{}(default交叉熵) "
        "\nStep:{} "
        "\n在Epoch开始验证保存模型:{"
        "}\n是否开启混合精度:{}\n".format(
            opt.epoches, opt.train_times, train_path, opt.batchSize,
            val_path, opt.batchSize, opt.test_path, opt.batchSize, opt.learning_rate, opt.enhance, opt.shuffle, opt.num_workers, opt.model_name,
            opt.FL, opt.step, opt.leastEpoch, opt.mixAcc))
    f.write("学习率策略:{}\n".format(opt.lr_scheduler))
    f.write("weight_decay:{}\n".format(opt.weight_decay))
    if opt.raw_image:
        f.write("使用原始图像raw_image:{}\n".format(opt.raw_image))
        f.write("等比缩放大小raw_size:{}\n".format(opt.raw_size))
    f.write("-----------初始化已完成...正在启动...-----------\n")


def draw_loss_acc(save_path: list, loss_list: list, acc_list: list):
    """
    绘制损失与准确率图像
    :param save_path: [存放图像的路径, k:标识符]
    :param loss_list: 损失曲线
    :param acc_list: 准确率曲线
    :return:
    """
    # 损失曲线
    path = save_path[0] + '/' + str(save_path[1])
    loss = np.array(loss_list)
    plt.plot(loss, 'g')
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.title('Loss Curve')
    plt.savefig(path + 'loss.png')
    plt.close()
    # 准确率曲线
    acc = np.array(acc_list)
    plt.plot(acc, 'r')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.title('Acc Curve')
    plt.savefig(path + 'acc.png')
    plt.close()
    # acc_loss混合曲线
    length = len(loss) // len(acc)
    # 累加的曲线
    loss_epoch = np.zeros_like(acc)
    # 间隔的曲线
    # loss_epoch = loss[::length]
    for i in range(len(acc)):
        loss_epoch[i] = np.sum(loss[i * length:i * length + length])
    # 把图像画在一个图像上
    # Create some mock data
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('acc', color=color)
    ax1.plot(acc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(loss_epoch, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(path + 'loss_acc.png')
    plt.close()


def finall_test_acc(best_k, best_model):
    """
    加载在验证集上最好的模型, 并且在测试集上的结果
    :param best_k:最佳的k值
    :param best_model:最佳的模型编号
    :return:
    """
    # 最佳模型路径
    best_model_path = ModelSave_list[best_k] + '/' + 'last' + str(best_model) + '.pth'
    print("测试模型: ", best_model_path)
    f.write("测试模型: {}\n".format(best_model_path))
    # 最终测试集路径
    test_path = opt.test_path
    print("测试集路径: ", test_path)
    f.write("测试集路径: {}\n".format(test_path))
    # 初始化并加载模型
    model = choseModel(opt.model_name)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    # 开启测试模式
    model.eval()
    # 将模型放入GPU中
    model.to(device)
    # 数据变换与训练保持相同
    transform = data_enhance()
    # 加载测试数据集
    dataset_test = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    # DataLoader
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchSize, num_workers=opt.num_workers,
                                 shuffle=opt.shuffle, pin_memory=True)
    # 如果要统计其它指标, 那么就统计下面的test_target和test_predict
    test_target = []  # 测试标签
    test_predict = []  # 预测结果
    with torch.no_grad():
        # 统计结果1
        for step, data in enumerate(dataloader_test):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predict = torch.argmax(outputs, dim=1)
            # 将标签放入对应的列表, 然后调用sklearn进行acc统计
            test_target += labels.cpu()  # gpu tensor先转换成cpu tensor 然后才可以转换成numpy
            test_predict += predict.cpu()
        sub_test_acc = accuracy_score(test_predict, test_target)
        print("最终测试集的子图准确率为:{}".format(sub_test_acc))
        f.write("最终测试集的子图准确率为:{}\n".format(sub_test_acc))
        print("")
        f.write("\n")

        # ----------------统计结果2: 以原图为单位----------------
        # 读取一个整的路径
        root_dir = test_path
        # 文件夹的类标签
        class_labels = dataset_test.classes
        # 对应标签转换
        class_change = dataset_test.class_to_idx
        # 标签对应类别转换
        labels_class = {}
        for k, v in class_change.items():
            labels_class[str(v)] = k

        # 重写的dataset
        G = GetMyData(root_dir, class_labels, transform, class_change)
        dataset = G.master()
        # 自定义读取图像
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, num_workers=opt.num_workers,
                                shuffle=opt.shuffle, pin_memory=True)
        #  记录土壤以整张大图为单位的识别准确率
        class_name_all = {}  # 整体预测结果
        for step, data in enumerate(dataloader):
            inputs, labels, flag_classNum = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predict = torch.argmax(outputs, dim=1)
            # 统计土壤大图的整体准确率 -> 2022年04月11日15:30:55下面这个循环, 其实还可以简化, 不过目前功能已经完成就暂时不去纠结了
            for i in range(len(labels)):
                img_name = flag_classNum[i]
                img_label = str(labels[i].cpu().numpy())
                img_predict = str(predict[i].cpu().numpy())
                img_key = img_name
                if img_key not in class_name_all:
                    null_dict = {}  # 初始化空字典
                    for c in class_labels:
                        null_dict[c] = 0
                    class_name_all[img_key] = null_dict  # 初始化
                    class_name_all[img_key][labels_class[img_predict]] += 1  # 对应位置+1
                else:
                    class_name_all[img_key][labels_class[img_predict]] += 1
        # 探讨一下上面的代码有无错误, 然后统计正确率
        # 2022年04月10日02:04:32, 我太困了 = = 没睡醒写得太繁琐了, 明天来修改代码! 这样的代码有点蠢 我明天再来修改 今天先跑起来 -> 2022年04月11日10:45:10现在来改
        # 大图在每个类的准确率
        class_all_acc = class_name_all.copy()
        for key in class_all_acc:
            key_v = class_all_acc[key]
            s = 0
            for v in key_v.values():
                s += v
            for k in key_v.keys():
                class_all_acc[key][k] = class_all_acc[key][k] / s
        print("原图整体测试信息:")
        print(class_name_all)
        print(class_all_acc)
        print("")
        f.write("原图整体测试信息:\n")
        f.write("{}\n".format(class_name_all))
        f.write("{}\n".format(class_all_acc))
        f.write("\n")
        # 输出大图的准确率
        total_true = 0
        origin_true = []  # 原图标签
        origin_predict = []  # 原图预测标签
        for key in class_all_acc:
            key_v = class_all_acc[key]
            a, b = list(key_v.keys()), list(key_v.values())
            max_class = b.index(max(b))
            true_class = class_change[key.split('_')[0]]
            # 大图的混淆矩阵
            origin_true.append(true_class)
            origin_predict.append(max_class)
            if max_class == true_class:
                total_true += 1
        # 初始化字典, 统计单张信息
        predict_num = {}
        predict_acc = {}
        # 遍历key生成新的预测结果子字典
        for key in class_name_all.keys():
            class_key = key.split('_')[0]
            predict_num[key] = int(class_name_all[key][class_key])
            predict_acc[key] = class_all_acc[key][class_key]
        # 平均准确率
        total_avg_acc = 0
        for value in predict_acc.values():
            total_avg_acc += value
        total_avg_acc = total_avg_acc / len(predict_acc)
        print("原图单张土壤图像信息:")
        print(predict_num)
        print(predict_acc)
        print("")
        f.write("原图单张土壤图像信息:\n")
        f.write("{}\n".format(predict_num))
        f.write("{}\n".format(predict_acc))
        f.write("\n")
        print("总的大图平均测试准确率:", total_avg_acc)
        print("预测正确的原图数为: {} / {}   准确率为: {:.4f}\n".format(total_true, len(class_all_acc),
                                                           total_true / len(class_all_acc)))
        f.write("总的大图平均测试准确率: {}\n".format(total_avg_acc))
        f.write("预测正确的原图数为: {} / {}   准确率为: {:.4f}\n".format(total_true, len(class_all_acc),
                                                             total_true / len(class_all_acc)))
        f.write("")
        # 将真实标签以及最终的预测结果统计在predict.txt中, 便于后续统计其它指标
        f2.write("\ntest_sub: {}\ntest_orginal: {}\ntest_all: {} / {}    {}\n".format(sub_test_acc, total_avg_acc, total_true,
                                                                                      len(class_all_acc), total_true / len(class_all_acc)))
        # 子图标签 子图预测  大图标签  大图预测
        np.save(varSave + '/predict.npy', np.array([test_target, test_predict, origin_true, origin_predict], dtype=object))
        # 全部预测信息(原图整体测试信息), 单张图像正确标签准确率(原图单张土壤图像信息)
        np.save(varSave + '/test_dict.npy', np.array([class_name_all, predict_acc]))


def test_row_image(best_k, best_model):
    """
    加载在验证集上最好的模型, 并且在测试集上的结果(原始图像)
    :param best_k:最佳的k值
    :param best_model:最佳的模型编号
    :return:
    """
    # 最佳模型路径
    best_model_path = ModelSave_list[best_k] + '/' + 'last' + str(best_model) + '.pth'
    print("测试模型: ", best_model_path)
    f.write("测试模型: {}\n".format(best_model_path))
    # 最终测试集路径
    test_path = opt.test_path
    print("测试集路径: ", test_path)
    f.write("测试集路径: {}\n".format(test_path))
    # 初始化并加载模型
    model = choseModel(opt.model_name)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    # 开启测试模式
    model.eval()
    # 将模型放入GPU中
    model.to(device)
    # 数据变换与训练保持相同
    transform = data_enhance()
    # 加载测试数据集
    dataset_test = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    # DataLoader
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchSize, num_workers=opt.num_workers,
                                 shuffle=opt.shuffle, pin_memory=True)
    # 如果要统计其它指标, 那么就统计下面的test_target和test_predict
    test_target = []  # 测试标签
    test_predict = []  # 预测结果
    with torch.no_grad():
        # 统计结果1
        for step, data in enumerate(dataloader_test):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predict = torch.argmax(outputs, dim=1)
            # 将标签放入对应的列表, 然后调用sklearn进行acc统计
            test_target += labels.cpu()  # gpu tensor先转换成cpu tensor 然后才可以转换成numpy
            test_predict += predict.cpu()
        sub_test_acc = accuracy_score(test_predict, test_target)
        print("最终原图测试集准确率为:{}".format(sub_test_acc))
        f.write("最终原图测试集准确率为:{}\n".format(sub_test_acc))
        print("")
        f.write("\n")
        print("test_row_image: {}".format(sub_test_acc))
        f2.write("\ntest_row_image: {}".format(sub_test_acc))
        np.save(varSave + '/predict.npy', np.array([test_target, test_predict]))


if __name__ == '__main__':
    # 超参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='/Users/yida/Desktop/val/val_3',
                        help='对该数据集进行K折交叉验证, 对每个验证集进行验证')
    parser.add_argument('--val_path', type=str, default='',
                        help='验证集路径, 为空时默认将验证集路径设为测试集路径, 否则同测试集路径')
    parser.add_argument('--test_path', type=str, default='/Users/yida/Desktop/val/val_3', help='模型最终测试的测试集路径')
    parser.add_argument('--epoches', type=int, default=1, help='模型训练epoch次数')
    parser.add_argument('--train_times', type=int, default=1, help='网络训练次数')
    parser.add_argument('--batchSize', type=int, default=64, help='模型训练的Batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='开启多线程, 默认为0, 容易出错')
    parser.add_argument('--shuffle', action='store_false', default=True, help='打乱数据排序')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='优化器学习率enta')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='正则化权重惩罚, 避免过拟合')
    parser.add_argument('--n_class', type=int, default=3, help='设置分类数')
    # 2021年12月15日23:02:33 修改工具类,保存模型和参数的路径,仅用一个参数来替代
    parser.add_argument('--Tools_path', type=str, default='./Tools', help='保存模型的地方, 保存训练图像及训练损失曲线等参数的地方')
    parser.add_argument('--enhance', action='store_true', default=False, help='是否进行数据增强, 默认关闭')
    parser.add_argument('--model_name', type=str, default='resnet', help='选择需要的训练模型, 注意导入model类, 全部用名字称呼')
    parser.add_argument('--FL', action='store_true', default=False, help='选择FocalLoss, 替代默认的交叉熵损失函数')
    parser.add_argument('--step', type=int, default=1, help='每隔多少轮输出一次训练信息, 训练和验证同step')
    parser.add_argument('--leastEpoch', type=float, default=0, help='当大于leastEpoch时才进行验证, 减少训练时间')
    parser.add_argument('--mixAcc', action='store_false', default=True,
                        help='是否开启混合精度:好处是能提升速度10-30%, 缺点是精度损失且仅能在cuda上操作')
    parser.add_argument('--lr_scheduler', type=str, default='None', help='默认为None不使用学习率策略, 否则根据名称使用学习率策略')
    parser.add_argument('--raw_image', action='store_true', default=False, help='默认为False, 开启时直接使用原图进行训练和测试, 会修改最后准确率统计的地方')
    parser.add_argument('--raw_size', type=int, default=600, help='默认为224, 当使用原图大小时, 将其缩放到固定大小')
    parser.add_argument('--testWith_raw', action='store_true', default=False, help='测试时, 同原图一样, 但是数据变换没有使用缩放')
    opt = parser.parse_args()
    Summary = []  # 存放k折交叉验证的结果
    ModelSave_list = []  # 全局变量, 用于记录存放的最佳模型地址
    ModelNum_list = []  # 全局变量, 用于记录最佳k折中的模型编号
    # 定义全局变量
    acc_global = []  # 统计训练n次的平均准确率, 这个不能放在全局, 不然统计的是很多次的, 有问题
    train_path = opt.train_path  # 训练集路径
    val_path = opt.test_path if len(opt.val_path) == 0 else opt.val_path  # 验证集路径, 当验证集为空时, 将验证集指向测试集
    modelSvae, varSave = initialization()  # 模型存储路径, 参数存储路径
    ModelSave_list.append(modelSvae)  # 记录最佳模型路径
    # 2022年03月13日09:17:24新建记事本, 记录全部输出
    with open(varSave + '/result.txt', 'w') as f:
        # 初始化
        start = time.time()
        print("-------------------------trainSigle数据集:{}-------------------------".format(
            opt.train_path.split('/')[-1]))
        print("")
        f.write("-------------------------trainSingle数据集:{}-------------------------\n".format(
            opt.train_path.split('/')[-1]))
        f.write("\n")
        # 获取数据集和验证集路径
        # 基本信息写入txt
        temp_write()
        for k in range(opt.train_times):
            # 重复训练times次
            print("-----------正在训练第{}/{}次-----------".format(k + 1, opt.train_times))
            f.write("-----------正在训练第{}/{}次-----------\n".format(k + 1, opt.train_times))
            main()
        acc_summarize = Summarize()  # 输出一个数据集多次实验的均值
        sMean, sStd = np.mean(acc_summarize), np.std(acc_summarize)  # 新增计算
        Summary.append(sMean)
        print("")
        f.write("\n")
        Summarize()  # 输出总结信息
        end = time.time()
        time_total = end - start
        print("消耗时间:{}秒, {}分".format(time_total, time_total / 60))
        f.write("消耗时间:{}秒, {}分\n".format(time_total, time_total / 60))
        print("----------------END----------------")
        f.write("----------------END----------------\n")
        f.close()

    # 存储最终结果 到KfoldSummary.txt中去
    np.savetxt(varSave + '/KFoldSummary.txt', acc_summarize, fmt='%.4f')  # 保存结果到txt
    with open(varSave + '/KFoldSummary.txt', 'a') as f2, open(varSave + '/result.txt',
                                                              'a') as f:  # 追加最终结果到txt以及KFoldSummary
        f2.write("{:.4f}±{:.4f}".format(np.mean(acc_summarize), np.std(acc_summarize)))  # 修改成n次的结果
        # 最终测试...加载在最好k折验证下进行训练的模型 在最终数据集上进行测试
        # 1. 找到最好验证集准确率下对应的模型k
        best_k = Summary.index(max(Summary))  # 确定在哪个k中最佳的均值模型
        best_model = np.argmax(ModelNum_list[best_k])  # 在确定的k折中, 确定其中的最佳模型
        print("验证集上的平均准确率为:{} 其最佳的模型编号为:{} 最佳准确率:{}".format(Summary[best_k], best_model,
                                                            ModelNum_list[best_k][best_model]))
        f.write("验证集上的平均准确率为:{} 其最佳的模型编号为:{} 最佳准确率:{}\n".format(Summary[best_k], best_model,
                                                                ModelNum_list[best_k][best_model]))
        # 测试的时候需要单独对每一张图像, 所以这个代码需要修改设计一下, 可能要重写dataloader -> 已完成
        # 2022年03月29日16:20:08下面代码修改完毕, 但是还未写入txt中, 接下来请修改 -> 已完成
        # |--------------------测试--------------------|
        if opt.raw_image:
            test_row_image(best_k, best_model)  # 原图
        elif opt.testWith_raw:
            test_row_image(best_k, best_model)  # 原图, 但不使用数据变换
        else:
            finall_test_acc(best_k, best_model)  # 子图
        f.write("测试已完成...\n")
        print("测试已完成...\n")
        # 关闭文件流
        f.close()
        f2.close()
        # |--------------------已完成--------------------|
