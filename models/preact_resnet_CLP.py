"""
This file is modified based on the following source:
link : https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
The original license is placed at the end of this file.

This file provide implementation of pre-activation ResNet.
Please note that this is different from default ResNet in pytorch, even thought the structure of file is quite similar.
And to adapt different image size, we replace the Avgpool2d with its adaptive version.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter

class BatchNorm2d_CLP(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.diff = 0.0
        self.diff_relative = 0.0
        self.act = 0
        self.act_clean = 0

    def forward(self, x):
        n, c = x.shape[:2]
        # print(n, c)
        output = super().forward(x)
        benign_x = output[:n // 2].reshape(n // 2, c, -1)
        trigger_x = output[n // 2:].reshape(n // 2, c, -1)
        #         benign_x = output[:n//2].reshape(n//2, -1)
        #         trigger_x = output[n//2:].reshape(n//2, -1)

        self.diff += ((benign_x - trigger_x).norm(2, -1).mean(0) / output.reshape(n, -1).norm(2, -1).mean(0)).detach()
        self.diff_relative += ((trigger_x-benign_x).sum(-1).mean(0) / output.reshape(n, -1).norm(2, -1).mean(0)).detach()
        return output

class Conv2d_CLP(nn.Conv2d):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.diff = 0.0
        self.diff_relative = 0.0
        self.act = 0
        self.act_clean = 0
    def forward(self, x):
        
        # print(n, c)
        output = super().forward(x)
        n, c = output.shape[:2]
        benign_x = output[:n // 2].reshape(n // 2, c, -1)
        trigger_x = output[n // 2:].reshape(n // 2, c, -1)
        #         benign_x = output[:n//2].reshape(n//2, -1)
        #         trigger_x = output[n//2:].reshape(n//2, -1)

        self.diff += ((benign_x - trigger_x).norm(2, -1).mean(0) / output.reshape(n, -1).norm(2, -1).mean(0)).detach()
        self.diff_relative += ((trigger_x-benign_x).sum(-1).mean(0) / output.reshape(n, -1).norm(2, -1).mean(0)).detach()
        return output

class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = BatchNorm2d_CLP(in_planes)
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = Conv2d_CLP(in_planes, planes,  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d_CLP(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = Conv2d_CLP(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.ind = None

        if stride != 1 or in_planes != self.expansion * planes:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            # )
            self.shortcut = nn.Sequential(
                Conv2d_CLP(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.ind is not None:
            out += shortcut[:, self.ind, :, :]
        else:
            out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.use_simple = False
        self.in_planes = 64
        self.conv1 = Conv2d_CLP(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, alpha=1.):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


def PreActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])


def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3])


def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3])
