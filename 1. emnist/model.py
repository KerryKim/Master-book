## import libraries
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from layer import *

## 네트워크 구축하기

'''
Image size is 28*28*1 pixels

Conv block order : Conv → BN → ReLu
Output image size : ((n+2p-f)/s) + 1
To keep same image size : kernel_size 1, padding 0 // k 3, p 1

pooling을 하는 이유는 차원을 줄여서 계산량을 감소하기 위함이다.
'''

# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # This example for 28*28*1 image size
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.softmax(self.out(x), dim=1)

        return x    # this output is probability

    # self.num_flat_features() 메서드는 input 텐서의 총 parameter 갯수이다.
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
