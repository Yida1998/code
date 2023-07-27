"""
Author: yida
Time is: 2022/1/7 19:49 
this Code: 重新实现真正的Resnet18, 同torch官方实现的model
不能重复的原因:
1.没有按照指定方法初始化参数
2.BN层指定初始化准确率也能提升1-2%
结果:现在能和官方的model获得相同准确率
很值得参考的博客https://blog.csdn.net/weixin_44331304/article/details/106127552?spm=1001.2014.3001.5501
"""
import os

import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BasicBlock(nn.Module):
    def __init__(self, in_channel, s):
        """
        基础模块, 共有两种形态, 1.s=1输入输出维度相同时 2.s=2特征图大小缩小一倍, 维度扩充一倍
        :param in_channel: 输入通道数维度
        :param s: s=1 不缩小 s=2 缩小尺度
        """
        super(BasicBlock, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(in_channel, in_channel * s, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * s, in_channel * s, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * s)
        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * s, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(in_channel * s)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2:  # 缩小
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, n_class, zero_init_residual=False):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channel=128, s=2),
            BasicBlock(in_channel=256, s=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channel=256, s=2),
            BasicBlock(in_channel=512, s=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_class)

        # 初始化参数 -> 影响准确率 7%
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 初始化BasicBlock -> 影响准确率 1-2%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    inputs = torch.rand(10, 3, 224, 224)
    model = ResNet18(n_class=9)
    print(model)
    outputs = model(inputs)
    print(outputs.shape)
