import logging
import os

import librosa
import numpy as np
import torch
from python_speech_features import fbank
from torch import float32
from tqdm import tqdm

import Config
import SilenceDetector
import Tools
from Tools import selectFiles

sampleRate = Config.getMainConfig("sampleRate")


# 音频端点检测
def VAD(audio, sampleRate):
    chunkSize = int(sampleRate * 0.05)  # 50ms
    index = 0
    sil_detector = SilenceDetector.SilenceDetector(15)
    nonSilenceAudio = []
    while index + chunkSize < len(audio):
        if not sil_detector.is_silence(audio[index: index + chunkSize]):
            nonSilenceAudio.extend(audio[index: index + chunkSize])
        index += chunkSize
    return np.array(nonSilenceAudio)


# 读取音频并进行静音处理
def readAudio(filePath):
    global sampleRate
    audio = librosa.load(filePath, sr=sampleRate, mono=True)[0]
    audio = VAD(audio.flatten(), sampleRate)
    # 时长低于2秒的，扩充到2秒，剩下的用0填充
    if len(audio) < 2 * sampleRate:
        audioTemp = [0] * sampleRate * 2
        for i in range(len(audio)):
            audioTemp[i] = audio[i]
        audio = np.array(audioTemp)
    return audio
    # if len(audio) < 2 * sampleRate:
    #     return False
    # else:
    #     return audio


# 数据归一化，或者叫做标准化，类似于正态分布标准化的流程
def normalizeData(data, standardDeviation=1e-12):
    return [(v - np.mean(v, axis=0)) / (np.std(v, axis=0) + standardDeviation) for v in data]


# 给出原生音频数据和采样率，提取filterBank数据
def extractFilterBankFeature(rawAudio, sampleRate):
    filterBankFeature = fbank(rawAudio, sampleRate, nfilt=64, winlen=0.025)[0]
    filterBankFeature = normalizeData(filterBankFeature)
    filterBankFeature = torch.as_tensor(filterBankFeature, dtype=float32)
    return filterBankFeature


# 给定语音路径，返回处理后的filterBankData
def processOneUtterance(utterancePath):
    global sampleRate
    rawAudio = readAudio(utterancePath)
    filterBankFeature = extractFilterBankFeature(rawAudio, sampleRate)
    return filterBankFeature


def getFilterBankDataFromAudio(audio):
    global sampleRate
    filterBankFeature = extractFilterBankFeature(audio, sampleRate)
    filterBankFeature = filterBankFeature.reshape(1, -1, 64)
    filterBankData = Tools.padFilterBankData(filterBankFeature)
    return filterBankData


def filterBankProcess(originalDataSetPath, filterBankDataSetPath):
    logging.info("提取filterBank特征中...")
    if not os.path.exists(filterBankDataSetPath):
        os.mkdir(filterBankDataSetPath)

    # 获取数据列表
    originalDataFileList = selectFiles(originalDataSetPath + "\\**\\*.flac", True)
    # 处理数据并将特征数据放在本地data\\filterBankData文件夹下
    # 当前正在处理的说话人id
    nowSpeaker = ""
    # 当前数据会存放的路径，取决于说话人id
    currentSavePath = filterBankDataSetPath + "\\" + nowSpeaker
    for i in tqdm(range(len(originalDataFileList))):
        fileName = originalDataFileList[i].split('\\')[-1]
        logging.info("文件名:" + fileName)
        speaker = fileName.split('-')[0]
        logging.info("说话人:" + speaker)
        if not speaker == nowSpeaker:
            nowSpeaker = speaker
            currentSavePath = filterBankDataSetPath + "\\" + nowSpeaker
            if not os.path.exists(currentSavePath):
                logging.info("未找到存储路径，重新创建:" + currentSavePath)
                os.mkdir(currentSavePath)
        logging.info("当前存储路径:" + currentSavePath)
        targetDataFilePath = currentSavePath + "\\" + fileName.split(".")[0] + ".pt"
        if not os.path.exists(targetDataFilePath):
            # 下面是特征提取的主要函数
            filterBankFeature = processOneUtterance(originalDataFileList[i])
            torch.save(filterBankFeature, targetDataFilePath)


def processData():
    filterBankDataSetPath = Config.getMainConfig("filterBankDataSetPath")
    originalDataSetPath = Config.getMainConfig("originalDataSetPath")
    filterBankProcess(originalDataSetPath, filterBankDataSetPath)


def getMean(filterBankDataSetPath, meanSavePath):
    filterBankDataPathList = selectFiles(filterBankDataSetPath + "\\**\\*.pt", True)
    totalSum = 0
    totalNum = 0
    mean = 0
    for filterBankDataPath in filterBankDataPathList:
        filterBankData = torch.load(filterBankDataPath)
        filterBankDataSum = torch.sum(filterBankData)
        filterBankDataShape = filterBankData.size()
        filterBankDataNum = 1
        for dim in filterBankDataShape:
            filterBankDataNum = filterBankDataNum * dim
        totalSum = totalSum + filterBankDataSum
        totalNum = totalNum + filterBankDataNum
        print("当前均值：", (totalSum / totalNum).item())
    torch.save(mean, meanSavePath)


def getStandardDeviation(filterBankDataSetPath, meanSavePath, standardDeviationSavePath):
    filterBankDataPathList = selectFiles(filterBankDataSetPath + "\\**\\*.pt", True)
    mean = torch.load(meanSavePath)
    totalSum = 0
    totalNum = 0
    standardDeviation = 0
    for filterBankDataPath in filterBankDataPathList:
        filterBankData = torch.load(filterBankDataPath)
        filterBankData = filterBankData.reshape(-1)
        filterBankDataNum = filterBankData.size()[0]
        for data in filterBankData:
            totalSum = totalSum + (data - mean) ** 2
        totalNum = totalNum + filterBankDataNum
        standardDeviation = totalSum / totalNum
        print("当前标准差：", (totalSum / totalNum).item())
    torch.save(standardDeviation, standardDeviationSavePath)


def processDataWithMeanAndStandardDeviation(filterBankDataSetPath, filterBankDataSetWithNormPath, mean,
                                            standardDeviation):
    filterBankDataPathList = selectFiles(filterBankDataSetPath + "\\**\\*.pt", True)
    for filterBankDataPath in filterBankDataPathList:
        filterBankData = torch.load(filterBankDataPath)
        filterBankData = (filterBankData - mean) / max(standardDeviation, 1e-12)
        speakerId = filterBankDataPath.split("\\")[-2]
        fileName = filterBankDataPath.split("\\")[-1]
        currentSavePath = filterBankDataSetWithNormPath + "\\" + speakerId
        if not os.path.exists(currentSavePath):
            logging.info("未找到存储路径，重新创建:" + currentSavePath)
            os.mkdir(currentSavePath)
        savePath = currentSavePath + "\\" + fileName
        torch.save(filterBankData, savePath)


if __name__ == '__main__':
    processData()
