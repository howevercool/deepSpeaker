import logging
import os
import random
import time
from glob import glob

import numpy as np
import torch

import Config
from Model import DeepSpeakerModel


def selectFiles(path, recursive):
    return glob(path, recursive=recursive)


def cutAudio(audio, numFrame):
    if audio.shape[0] > numFrame:
        startIndex = random.randint(0, audio.shape[0] - numFrame)
        audio = audio[startIndex: numFrame + startIndex]
    else:
        padLength = numFrame - len(audio)
        padZeros = np.zeros((padLength, 64))
        audio = np.concatenate((audio, padZeros), axis=0)
    return audio


# 读取filterBank信息
def getFilterBankData(filterBankDataFilePath):
    filterBankData = torch.load(filterBankDataFilePath).cuda()
    filterBankData = cutAudio(filterBankData, Config.getMainConfig("numFrame"))
    return filterBankData


# 给定一个filterBankData, 形状是帧数 * 64，填充帧数到16的倍数
def padFilterBankData(filterBankData):
    length = len(filterBankData)
    padLength = 0
    while not (length + padLength) % 16 == 0:
        padLength += 1
    padZeros = torch.zeros((padLength, 64))
    filterBankData = torch.cat((filterBankData, padZeros), 0)
    return filterBankData


# def getAllFilterBankData(filterBankDataFilePath):
#     filterBankData = torch.load(filterBankDataFilePath)
#     length = len(filterBankData)
#     padLength = 0
#     while not (length + padLength) % 16 == 0:
#         padLength += 1
#     padZeros = np.zeros((padLength, 64, 1))
#     filterBankData = np.concatenate((filterBankData, padZeros), axis=0)
#     filterBankData = filterBankData.reshape(1, -1, 64)
#     filterBankData = torch.as_tensor(filterBankData, dtype=torch.float).cuda()
#     return filterBankData


def getFilterBankDataFromPath(filterBankDataFilePath):
    filterBankData = torch.load(filterBankDataFilePath)
    filterBankData = padFilterBankData(filterBankData)
    return filterBankData


# 对于输出的embed计算相似度
def getSimilarity(embed1, embed2):
    return torch.matmul(embed1, embed2).item()


# 获得说话人的嵌入
def getSpeakerEmbed(speakerId):
    return torch.load(os.path.join(Config.getMainConfig("speakerEmbedSavePath"), speakerId + ".pt"))


# 加载模型
def loadModelWithEval():
    model = DeepSpeakerModel().cuda()
    modelPath = Config.getMainConfig("modelPath")
    loadedParas = torch.load(modelPath)
    model.load_state_dict(loadedParas["modelStateDict"])
    model.eval()
    if "threshold" in loadedParas:
        threshold = loadedParas["threshold"]
    else:
        print("未找到相似度阈值，使用默认阈值")
        threshold = Config.getMainConfig("threshold")
    return model, threshold


# 将信息保存在文件内
def saveInformation(information, filePath):
    with open(filePath, 'a+') as file:
        file.write(str(information))
        file.write("\n")


# 检查文件夹内文件数量是否超过了上限，如果超过了就删除旧的文件
def checkFileNumAndDeleteMore(folderPath, numLimit):
    # 检查是否满了，如果数量满了，就去掉最老的一个
    fileList = os.listdir(folderPath)
    if len(fileList) >= numLimit:
        fileList = sorted(fileList, key=lambda x: os.path.getmtime(os.path.join(folderPath, x)))

    # 在文件远多于5个时，需要多次删除文件，这个下标指向没有被删除的最老的文件的下标
    fileIndex = 0
    # 开始删除超出数量上线的文件
    while len(fileList) - fileIndex > numLimit - 1:
        oldestFilePath = os.path.join(folderPath, fileList[fileIndex])
        os.remove(oldestFilePath)
        fileIndex = fileIndex + 1


# 获取文件夹内最新的文件的文件名
def getNewestFile(folderPath):
    fileList = os.listdir(folderPath)
    if len(fileList) == 0:
        print("目标文件夹内没有文件:", folderPath)
        exit(-1)
    fileList = sorted(fileList, key=lambda x: os.path.getmtime(os.path.join(folderPath, x)))
    return fileList[-1]


# 显示当前占用显存
def printVideoMemoryUsed():
    print("当前占用显存: %.5fGB " % (torch.cuda.memory_reserved() / 1024 / 1024 / 1024))


class TimeMeasure:
    def __init__(self):
        self.startTime = time.time()

    def start(self):
        self.startTime = time.time()

    def printTime(self):
        endTime = time.time()
        spendTime = time.time() - self.startTime
        print("耗时:{:.4f}秒".format(spendTime))
        print('\n')
        self.startTime = endTime
        return spendTime

    def printTimeWithMessage(self, message):
        endTime = time.time()
        spendTime = time.time() - self.startTime
        print(message)
        print("耗时:{:.4f}秒".format(spendTime))
        print('\n')
        self.startTime = endTime
        return spendTime

    def getSpendTime(self):
        endTime = time.time()
        spendTime = time.time() - self.startTime
        self.startTime = endTime
        return spendTime
