"""
Author: yida
Time is: 2022/3/29 21:25
this Code: 获取数据集的类标签
使用方法:
1.输出路径
2.把输出结果复制 -> 然后替换子图切割的类标签
"""
import torchvision


if __name__ == '__main__':
    path = '/Users/yida/Desktop/最终数据集/test'
    dataset = torchvision.datasets.ImageFolder(root=path)
    class_label = dataset.classes
    print(class_label)
    class_idx = dataset.class_to_idx
    print(class_idx)
