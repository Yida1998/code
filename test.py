"""
Author: yida
Time is: 2021/12/12 16:15 
this Code: 
"""
import torch
import torch.nn as nn
from matplotlib.font_manager import _rebuild
from matplotlib import font_manager
_rebuild()


class ModelNet(nn.Module):
    def __init__(self, n_class):
        super(ModelNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.n_class = n_class

    def forward(self, x):
        x = self.conv1(x)
        print("卷积层的输出:", x.shape)
        x = self.pool(x)
        print("池化层的输出:", x.shape)
        return x


if __name__ == '__main__':
    inputs = torch.rand(10, 3, 224, 224)
    print("输入:", inputs.shape)
    model = ModelNet(n_class=10)
    outputs = model(inputs)
    print("输出:", outputs.shape)

    a = sorted([f.name for f in font_manager.fontManager.ttflist])
    for i in a:
        print(i)