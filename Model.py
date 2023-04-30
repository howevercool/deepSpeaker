import collections

import torch
import torch.nn as nn


class ClippedReLu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, inputTensor):
        x = self.relu(inputTensor)
        x = torch.clamp(x, max=20.0)
        return x


class ResBlock(nn.Module):
    def __init__(self, numFilter, kernelSize=3):
        super().__init__()
        self.resBlock = nn.Sequential(
            nn.Conv2d(numFilter, numFilter, kernelSize, padding=kernelSize // 2),
            nn.BatchNorm2d(numFilter),
            ClippedReLu(),
            nn.Conv2d(numFilter, numFilter, kernelSize, padding=kernelSize // 2),
            nn.BatchNorm2d(numFilter)
        )
        self.clippedRelu = ClippedReLu()

    def forward(self, inputTensor):
        outputTensor = self.resBlock(inputTensor)
        outputTensor += inputTensor
        outputTensor = self.clippedRelu(outputTensor)
        return outputTensor


class ConvAndRes(nn.Module):
    def __init__(self, inChannel, outChannel):
        super().__init__()
        # 不带命名版本
        self.convAndRes = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(outChannel),
            ClippedReLu(),
            ResBlock(outChannel),
            ResBlock(outChannel),
            ResBlock(outChannel)
        )

    def forward(self, inputTensor):
        return self.convAndRes(inputTensor)


class DeepArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.deepArchitecture = nn.Sequential(
            ConvAndRes(1, 64),
            ConvAndRes(64, 128),
            ConvAndRes(128, 256),
            ConvAndRes(256, 512)
        )

    def forward(self, x):
        return self.deepArchitecture(x)


class DeepSpeakerModel(nn.Module):
    def __init__(self):
        super(DeepSpeakerModel, self).__init__()
        self.netModuleDict = nn.ModuleDict({
            "deepArchitecture": DeepArchitecture(),
            "affine": nn.Linear(2048, 512)
        })

    def forward(self, x):
        numFrame = x.shape[2]
        x = self.netModuleDict["deepArchitecture"](x)
        x = x.view(-1, int(numFrame / 16), 2048)
        x = x.mean(1)
        x = self.netModuleDict["affine"](x)
        y = x.clone()
        # 下面是长度归一化层
        # for i in range(len(x)):
        #     l2norm = torch.norm(x[i])
        #     y[i] = x[i] / l2norm
        l2norm = torch.norm(x, dim=1)
        for i in range(len(x)):
            y[i] = x[i] / l2norm[i]
        return y


# 带有softmax分类层的Deep speaker模型，用来进行预训练
class SoftmaxDeepSpeakerModel(nn.Module):
    def __init__(self, numFrame, numSpeaker):
        # TODO 应当使用更高级的模型定义方法
        super(SoftmaxDeepSpeakerModel, self).__init__()
        self.numFrame = numFrame
        self.netModuleDict = nn.ModuleDict({
            "deepArchitecture": DeepArchitecture(),
            "affine": nn.Linear(2048, 512),
            "dropout": nn.Dropout(0.5),
            "affine2": nn.Linear(512, numSpeaker)
        })

    def forward(self, x):
        x = self.netModuleDict["deepArchitecture"](x)
        x = x.view(-1, int(self.numFrame / 16), 2048)
        x = x.mean(1)
        x = self.netModuleDict["affine"](x)
        x = self.netModuleDict["dropout"](x)
        x = self.netModuleDict["affine2"](x)
        return x
