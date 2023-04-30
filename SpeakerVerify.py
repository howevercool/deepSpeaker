import logging
import os
import random
import shutil

import numpy as np
import torch
from tqdm import tqdm

import Config
import Tools
from ProcessData import readAudio, extractFilterBankFeature, processOneUtterance
from Tools import selectFiles, cutAudio, getSimilarity, getSpeakerEmbed, loadModelWithEval


def speakerRegisterUseFilterBankData(deepSpeaker, utteranceFilterBank, speakerId):
    embed = deepSpeaker(utteranceFilterBank)
    embed = embed.t()
    embed = torch.mm(embed, torch.ones(len(utteranceFilterBank), 1).cuda()).t()
    embed = torch.squeeze(embed)
    embed = embed / torch.norm(embed)
    print(torch.norm(embed))
    torch.save(embed, os.path.join(Config.getMainConfig("speakerEmbedSavePath"), speakerId + ".pt"))
    return embed


def speakerRegisterWithOneUtterance(deepSpeaker, utteranceFilterBank, speakerId):
    embed = deepSpeaker(utteranceFilterBank)
    embed = torch.squeeze(embed)
    torch.save(embed, os.path.join(Config.getMainConfig("speakerEmbedSavePath"), speakerId + ".pt"))
    return embed


def speakerRegister(deepSpeaker, speakerId, utterancePathList):
    utteranceFilterBank = []
    # 获取数据
    for i in range(len(utterancePathList)):
        sampleRate = Config.getMainConfig("sampleRate")
        rawAudio = readAudio(utterancePathList[i], sampleRate)
        filterBankFeature = extractFilterBankFeature(rawAudio, sampleRate)
        filterBankFeature = cutAudio(filterBankFeature, Config.getMainConfig("numFrame"))
        filterBankFeature = filterBankFeature.reshape(1, -1, 64)
        filterBankFeature = torch.as_tensor(filterBankFeature, dtype=torch.float).cuda()
        utteranceFilterBank.append(filterBankFeature)
    utteranceFilterBank = torch.stack(utteranceFilterBank)
    return speakerRegisterUseFilterBankData(deepSpeaker, utteranceFilterBank, speakerId)


def speakerRegisterWithFilterBankDataPath(deepSpeaker, speakerId, utterancePathList):
    utteranceFilterBank = []
    # 获取数据
    for i in utterancePathList:
        filterBankFeature = Tools.getFilterBankData(i)
        utteranceFilterBank.append(filterBankFeature)
    utteranceFilterBank = torch.stack(utteranceFilterBank)
    return speakerRegisterUseFilterBankData(deepSpeaker, utteranceFilterBank, speakerId)


def speakerVerify(speakerEmbed, utterancePath, deepSpeaker, threshold):
    # 获取语音的filterBank信息
    filterBankData = processOneUtterance(utterancePath)
    filterBankData = Tools.padFilterBankData(filterBankData).cuda()
    return speakerVerifyWithFilterBankData(speakerEmbed, filterBankData, deepSpeaker, threshold)


def speakerVerifyWithFilterBankDataFile(speakerEmbed, utterancePath, deepSpeaker, threshold):
    filterBankFeature = Tools.getFilterBankDataFromPath(utterancePath).cuda()
    return speakerVerifyWithFilterBankData(speakerEmbed, filterBankFeature, deepSpeaker, threshold)


def speakerVerifyWithFilterBankData(speakerEmbed, filterBankData, deepSpeaker, threshold):
    filterBankData = filterBankData.unsqueeze(0)
    filterBankData = filterBankData.unsqueeze(0)
    utteranceEmbed = deepSpeaker(filterBankData)
    utteranceEmbed = torch.squeeze(utteranceEmbed)
    similarity = getSimilarity(speakerEmbed, utteranceEmbed)
    if similarity > threshold:
        return True
    else:
        return False


def processData():
    # 获取数据列表
    utterancePath = "D:\\pythonProject\\DeepSpeaker\\dataSet\\ForTest\\61"
    filterBankDataSavePath = "D:\\pythonProject\\DeepSpeaker\\dataSet\\ForTest\\FilterBank"
    originalTrainDataFileList = selectFiles(utterancePath + "\\**\\*.flac", True)
    # 处理数据并将特征数据放在本地data\\filterBankData文件夹下
    # 当前数据会存放的路径，取决于说话人id
    for i in range(len(originalTrainDataFileList)):
        fileName = originalTrainDataFileList[i].split('\\')[-1]
        logging.info("文件名:" + fileName)
        speaker = fileName.split('-')[0]
        targetDataFilePath = filterBankDataSavePath + "\\" + fileName.split(".")[0] + ".npy"
        if not os.path.exists(targetDataFilePath):
            # 下面是特征提取的主要函数
            # TODO 应该将特征提取写的更分离一点，写成四个函数，分别是读取音频，VAD，特征提取，归一化
            sampleRate = Config.getMainConfig("sampleRate")
            rawAudio = readAudio(originalTrainDataFileList[i], sampleRate)
            filterBankFeature = extractFilterBankFeature(rawAudio, sampleRate)
            filterBankFeature = cutAudio(filterBankFeature, Config.getMainConfig("numFrame"))
            filterBankFeature = filterBankFeature.reshape(1, -1, 64)
            np.save(targetDataFilePath, filterBankFeature)


# 使用filterBank进行注册和测试
def testModelWithFilterBank():
    with torch.no_grad():
        deepSpeaker, threshold = loadModelWithEval()
        speakerId = '19'
        testSetPath = Config.getMainConfig("testSetPath")
        # threshold = 0.92
        print("阈值是否在训练:", threshold)
        speakerDataPath = os.path.join(testSetPath, speakerId)
        allUtteranceList = selectFiles(speakerDataPath + "\\**\\*.pt", True)
        registerUtteranceList = random.sample(allUtteranceList, 1)
        utteranceFilterBankData = Tools.getFilterBankDataFromPath(registerUtteranceList[0]).cuda()
        utteranceFilterBankData = torch.unsqueeze(utteranceFilterBankData, 0)
        utteranceFilterBankData = torch.unsqueeze(utteranceFilterBankData, 0)
        speakerRegisterWithOneUtterance(deepSpeaker, utteranceFilterBankData, speakerId)

        # 下面开始验证
        speakerEmbed = getSpeakerEmbed(speakerId)
        speakerList = os.listdir(testSetPath)
        totalRight = 0
        numUtterance = 0
        for speaker in speakerList:
            utterancePath = os.path.join(testSetPath, speaker)
            verifyUtteranceList = selectFiles(utterancePath + "\\**\\*.pt", True)
            numOfUtterance = len(verifyUtteranceList)
            numUtterance += numOfUtterance
            numTrue = 0
            for i in verifyUtteranceList:
                result = speakerVerifyWithFilterBankDataFile(speakerEmbed, i, deepSpeaker, threshold)
                if result:
                    numTrue += 1
            if speaker == speakerId:
                print("说话人:", speaker, "通过率:", numTrue / numOfUtterance)
                totalRight += numTrue
            else:
                print("说话人:", speaker, "拒绝率:", 1 - numTrue / numOfUtterance)
                totalRight += numOfUtterance - numTrue
        accuracy = totalRight/numUtterance
        print("平均正确率:", accuracy)


# 使用filterBank进行注册和测试
def testModelWithUtterance():
    with torch.no_grad():
        deepSpeaker, threshold = loadModelWithEval()
        speakerId = '60'
        registerUtteranceSetPath = Config.getMainConfig("registerUtteranceSetPath")
        # threshold = 0.92
        print("阈值:", threshold)
        print("模型是否在训练: ", deepSpeaker.training)
        speakerDataPath = os.path.join(registerUtteranceSetPath, speakerId)
        allUtteranceList = selectFiles(speakerDataPath + "\\**\\*.flac", True)
        registerUtteranceList = random.sample(allUtteranceList, 1)
        utteranceFilterBankData = processOneUtterance(registerUtteranceList[0])
        utteranceFilterBankData = Tools.padFilterBankData(utteranceFilterBankData)
        utteranceFilterBankData = torch.unsqueeze(utteranceFilterBankData, 0)
        utteranceFilterBankData = torch.unsqueeze(utteranceFilterBankData, 0).cuda()
        speakerEmbed = speakerRegisterWithOneUtterance(deepSpeaker, utteranceFilterBankData, speakerId)

        # 下面开始验证
        speakerList = os.listdir(registerUtteranceSetPath)
        totalRight = 0
        numUtterance = 0
        for speaker in speakerList:
            utterancePath = os.path.join(registerUtteranceSetPath, speaker)
            verifyUtteranceList = selectFiles(utterancePath + "\\**\\*.flac", True)
            numOfUtterance = len(verifyUtteranceList)
            numUtterance += numOfUtterance
            numTrue = 0
            for i in tqdm(verifyUtteranceList):
                result = speakerVerify(speakerEmbed, i, deepSpeaker, threshold)
                if result:
                    numTrue += 1
            if speaker == speakerId:
                print("说话人:", speaker, "通过率:", numTrue / numOfUtterance)
                totalRight += numTrue
            else:
                print("说话人:", speaker, "拒绝率:", 1 - numTrue / numOfUtterance)
                totalRight += numOfUtterance - numTrue
        accuracy = totalRight/numUtterance
        print("平均正确率:", accuracy)


# 用文献的方式测试模型
def testModelPaper():
    with torch.no_grad():
        deepSpeaker, threshold = loadModelWithEval()
        # threshold = 0.92
        print("阈值:", threshold)
        print("模型状态: ", deepSpeaker.training)

        registerUtteranceSetPath = Config.getMainConfig("registerUtteranceSetPath")
        speakerIdList = os.listdir(registerUtteranceSetPath)
        numSpeaker = 0
        numTrue = 0
        for speakerId in tqdm(speakerIdList):
            utteranceSetPath = os.path.join(registerUtteranceSetPath, speakerId)
            utterancePathList = selectFiles(utteranceSetPath + "\\**\\*.flac", True)
            registerUtterancePath = random.sample(utterancePathList, 1)[0]
            utteranceFilterBankData = processOneUtterance(registerUtterancePath)
            utteranceFilterBankData = Tools.padFilterBankData(utteranceFilterBankData)
            utteranceFilterBankData = torch.unsqueeze(utteranceFilterBankData, 0)
            utteranceFilterBankData = torch.unsqueeze(utteranceFilterBankData, 0).cuda()
            speakerEmbed = speakerRegisterWithOneUtterance(deepSpeaker, utteranceFilterBankData, speakerId)

            verifyUtterancePath = random.sample(utterancePathList, 1)[0]
            utteranceFilterBankData = processOneUtterance(verifyUtterancePath)
            utteranceFilterBankData = Tools.padFilterBankData(utteranceFilterBankData).cuda()
            result = speakerVerifyWithFilterBankData(speakerEmbed, utteranceFilterBankData, deepSpeaker, threshold)
            if result:
                numTrue = numTrue + 1
            numSpeaker = numSpeaker + 1
        print("准确率: ", numTrue/numSpeaker)


def generateRegisterSet():
    utteranceSetPath = "D:\\Graduation Project\\data\\LibriSpeech ASR corpus\\data\\LibriSpeech\\train-clean-100"
    registerDataSetPath = Config.getMainConfig("registerDataSetPath")
    speakerIdList = os.listdir(utteranceSetPath)
    registerSpeakerIdList = random.sample(speakerIdList, 60)
    for speakerId in tqdm(registerSpeakerIdList):
        shutil.copytree(os.path.join(utteranceSetPath, speakerId), os.path.join(registerDataSetPath, speakerId))


if __name__ == '__main__':
    testModelPaper()
