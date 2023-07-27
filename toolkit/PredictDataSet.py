"""
Author: yida
Time is: 2022/3/28 20:04 
this Code: 重写dataset类, 让其能够在预测的时候分别统计每一张大图的正确率及整体正确率(仅仅用在最终预测的时候).
注意事项:
1.记得输入正确的类文件名称标签
2.记得修改原始dataset所对应默认的标签

代码测试没有问题, 现在开始修改代码的细节

2022年03月28日21:51:01
正在新增测试代码
问题:
1.变量名太长的时候 其实也不太清晰
"""
import os

import numpy as np
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, root_dir, root_label, transfrom, class_change):
        self.root = root_dir
        self.label = root_label
        self.transform = transfrom
        self.class_change = class_change
        path = os.path.join(root_dir, root_label)
        self.file = os.listdir(path)
        # 移除Mac下的隐藏文件
        if '.DS_Store' in self.file:
            self.file.remove('.DS_Store')
        self.len = len(self.file)

    def __getitem__(self, idx):
        img_name = self.file[idx]
        img_item_path = os.path.join(self.root, self.label, img_name)
        # 打开图像
        img = Image.open(img_item_path)
        # 必须要转换成tensor
        img = self.transform(img)
        # 获取图像标签
        label = self.label
        label = self.change_label(label)  # 修改lable和我代码所默认的标签所对应
        # 转换成与dataset相同的数据类型torch.int64
        label = torch.tensor(label, dtype=torch.int64)
        # 将img_class_imgNum作为唯一标识符[类别_图像编号]
        img_flag = img_name.split('.')[0].split('_')[1:]
        img_class_imgNum = img_flag[0] + '_' + img_flag[1]
        return img, label, img_class_imgNum

    def __len__(self):
        """
        getitem - > 获取的索引就是0,len
        :return:
        """
        return self.len

    def change_label(self, label):
        """
        修改对应标签
        :param label:
        :return:
        """
        class_label = self.class_change
        new_label = class_label[label]
        return new_label


class GetMyData():
    def __init__(self, root_path: str, class_label: tuple, my_transfrom, class_change: dict):
        self.root = root_path
        self.label = class_label
        self.transfrom = my_transfrom
        self.class_change = class_change

    def master(self):
        """
        调用函数, 主要目的就是把所有dataset合并成一个dataset
        :return:
        """
        class_label = self.label
        root = self.root
        transfrom = self.transfrom
        class_change = self.class_change
        flag = True
        for item in class_label:
            dataset = MyData(root, item, transfrom, class_change)  # 获取包含路径的dataset
            # flag作为标识符, 为了跳过第一次, 其中第一次不进行合并
            if flag:
                flag = False
            else:
                dataset = self.data_cat(temp, dataset)
            temp = dataset
        return dataset

    def data_cat(self, data1, data2):
        """
        拼接数据集
        *数据集可以直接相加 dataset1 + dataset2*: 简单理解就是将返回的数据集进行合并
        :return:
        """
        data = data1 + data2
        return data


if __name__ == '__main__':
    # 输入图像变换形式
    transfrom = torchvision.transforms.ToTensor()
    # 读取一个整的路径
    root_dir = "/Users/yida/Desktop/train/TrainSet1_325_162"
    # 文件夹的类标签
    class_labels = ['0', '1', '3', '4', '6', '7', '8', '9', '10']
    change_labels = {'0': 0, '1': 1, '3': 2, '4': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8}
    G = GetMyData(root_dir, class_labels, transfrom, change_labels)
    dataset = G.master()
    # 自定义读取图像
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    # 2022年03月28日21:06:21 新增 -> 实现自定义预测
    class_num_real = {}  # 真实标签字典记录
    class_num_predict = {}  # 预测标签字典记录
    for step, data in enumerate(dataloader):
        inputs, label, flag_classNum = data
        predict = torch.randint(0, 9, label.shape)
        print(inputs.shape)
        print(label)
        print(predict)
        print(flag_classNum)
        print("================================")
        # 把真实标签, 放入字典中去
        for i in flag_classNum:
            if i not in class_num_real:
                class_num_real[i] = 1
            else:
                class_num_real[i] += 1
        # 把预测正确的标签, 放入预测字典中去
        flag = label == predict  # 标记预测正确的位置[true...false...]
        true_predict = predict[flag]  # 将预测正确的标签拿出来, 此时是预测正确的类别
        print(true_predict)
        flag_classNum = np.array(flag_classNum)  # 转换成numpy 的字符类型, 方便切片
        true_flag_classNum = flag_classNum[flag]  # 预测正确的唯一标识符
        # 将预测正确的子图, 统计到预测标记字典中去
        if len(true_flag_classNum) > 0:
            for j in true_flag_classNum:
                if j not in class_num_predict:
                    class_num_predict[j] = 1
                else:
                    class_num_predict[j] += 1
    # 探讨一下上面的代码有无错误, 然后统计正确率
    print(class_num_real)
    print(class_num_predict)
    # 统计大图分类正确率
    class_result = {}  # 存储分类结果
    for k in class_num_real.keys():
        if k in class_num_predict:
            class_result[k] = class_num_predict[k] / class_num_real[k]
        else:
            class_result[k] = 0
    print("每张土壤图像测试集的准确率为:", class_result)
    # 平均准确率
    total_avg_acc = 0
    for v in class_result.values():
        total_avg_acc += v
    total_avg_acc = total_avg_acc / len(class_result)
    print("总的平均测试准确率:", total_avg_acc)
