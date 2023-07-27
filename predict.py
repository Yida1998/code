"""
Author: yida
Time is: 2022/4/10 00:07 
this Code: 加载模型, 在测试集上预测结果
1.输入模型路径
2.输入测试集路径
3.得到统计结果
"""
import argparse

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.models import alexnet  # alexnet
from torchvision.models import resnet18  # resnet18

from Model.Resnet18 import ResNet18  # resnet
from Model.ResnetSe18 import ResNetSe18  # resnet-se
from Model.testModel import TestModel  # testnet
from toolkit.PredictDataSet import GetMyData  # 测试阶段自定义数据集

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_enhance():
    """
    是否进行数据增强
    :return:
    """
    if opt.enhance:
        data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 以概率P水平翻转
            transforms.RandomVerticalFlip(p=0.5),  # 以概率P垂直翻转  都是翻转操作  旋转会有黑边,所以展示不考虑
            transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0),  # 亮度随机增强百分之50
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.55, 0.41, 0.38), (0.15, 0.15, 0.15)),
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
    else:
        print("Which Modul are you need...")
        f.write("Which Modul are you need...\n")


def finall_test_acc(model_path, test_path):
    """
    加载在验证集上最好的模型, 并且在测试集上的结果
    :param model_path: 模型参数
    :param test_path: 测试集路径
    :return:
    """
    # 最佳模型路径
    print("测试模型: ", model_path)
    # 最终测试集路径
    print("测试集路径: ", test_path)
    # 初始化并加载模型
    model = choseModel(opt.model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
        print("")

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
        # 输出大图的准确率
        total_true = 0
        for key in class_all_acc:
            key_v = class_all_acc[key]
            a, b = list(key_v.keys()), list(key_v.values())
            max_class = b.index(max(b))
            true_class = class_change[key.split('_')[0]]
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
        print("总的大图平均测试准确率:", total_avg_acc)
        print("预测正确的原图数为: {} / {}\n".format(total_true, len(class_all_acc)))
        # 将真实标签以及最终的预测结果统计在predict.txt中, 便于后续统计其它指标
        print("\ntest_sub: {}\ntest_orginal: {}\ntest_all: {} / {}\n".format(sub_test_acc, total_avg_acc, total_true,
                                                                             len(class_all_acc)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 设置路径
    parser.add_argument('--model_path', type=str,
                        default='/Users/yida/Desktop/0824/TrainSet2_224_dropm0.0_adaption_1.0/2138/model/last0.pth', help='训练集路径')
    parser.add_argument('--test_path', type=str, default='/Users/yida/Desktop/0824/TrainSet2_224_dropm0.0_adaption_1.0/2138/model/val_/TrainSet2_224_dropm0.0_adaption_1.0', help='测试集路径')
    # 设置超参数
    parser.add_argument('--batchSize', type=int, default=64, help='模型训练的Batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_worker')
    parser.add_argument('--n_class', type=int, default=11, help='设置分类数')
    parser.add_argument('--enhance', action='store_true', default=False, help='是否进行数据增强, 默认关闭')
    parser.add_argument('--model_name', type=str, default='resnet', help='选择需要的训练模型, 注意导入model类, 全部用名字称呼')
    # 启动开关
    parser.add_argument('--shuffle', action='store_false', default=True, help='是否打乱数据集, 默认为True, 调用--shuffle开启为False')
    opt = parser.parse_args()  # 实例化
    model_path_ = opt.model_path
    test_path_ = opt.test_path
    finall_test_acc(model_path_, test_path_)
