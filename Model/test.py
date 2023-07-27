"""
Author: yida
Time is: 2022/1/4 17:28 
this Code: 官方model
"""
import os

import torch
from torchvision.models import resnet18

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':
    model = resnet18(pretrained=False, num_classes=9)
    model.load_state_dict(torch.load('/Users/yida/Desktop/model/4类土种识别/土壤识别模型/best0.pth', map_location='cpu'))
    print(model)
