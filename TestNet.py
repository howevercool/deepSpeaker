import random

import numpy as np
import torch

import Config
from Tools import getSimilarity, getFilterBankData


# 获取对于特定说话人的一组测试用数据
def getTestDataForOneSpeaker(speakerAndUtteranceDict, speaker):
    anchorPositivePair = random.sample(speakerAndUtteranceDict[speaker], 2)
    numNegativeInTest = Config.getMainConfig("numNegativeInTest")
    allNegativeList = []
    for i in speakerAndUtteranceDict:
        if not i == speaker:
            allNegativeList.extend(speakerAndUtteranceDict[i])
    negativeList = random.sample(allNegativeList, numNegativeInTest)
    # 开始读取数据
    filterBankDataList = []
    labelList = []
    filterBankDataList.append(torch.unsqueeze(getFilterBankData(anchorPositivePair[0]), 0))
    filterBankDataList.append(torch.unsqueeze(getFilterBankData(anchorPositivePair[1]), 0))
    labelList.append(1)
    for i in negativeList:
        filterBankDataList.append(torch.unsqueeze(getFilterBankData(i), 0 ))
        labelList.append(0)
    filterBankDataList = torch.stack(filterBankDataList).cuda()
    return filterBankDataList, labelList
    # return anchorFilterBankData, positiveFilterBankData, negativeFilterBankData


def getEER(similarityList, realLabelList):
    thresholdList = np.arange(0, 1.0, 0.01)
    bestEER = 1
    bestDifference = 1.0
    bestThreshold = 0
    bestFAR = 0
    bestFRR = 0
    for threshold in thresholdList:
        predictLabelList = []
        for i in similarityList:
            if i > threshold:
                predictLabelList.append(1)
            else:
                predictLabelList.append(0)
        # 相同说话人且预测成功的
        numTrueRefuse = 0
        numSameSpeaker = 1
        numDifferentSpeaker = Config.getMainConfig("numNegativeInTest")
        numFalseAccept = 0
        for i in range(len(predictLabelList)):
            if predictLabelList[i] == 1 and realLabelList[i] == 0:
                numFalseAccept += 1
            if predictLabelList[i] == 0 and realLabelList[i] == 1:
                numTrueRefuse += 1
        FRR = numTrueRefuse / numSameSpeaker
        FAR = numFalseAccept / numDifferentSpeaker
        if abs(FAR - FRR) <= bestDifference:
            bestDifference = abs(FAR - FRR)
            bestEER = (FAR + FRR) / 2
            bestThreshold = threshold
            bestFRR = FRR
            bestFAR = FAR
    return bestEER, bestThreshold, bestFRR, bestFAR


def testNet(model, speakerAndUtteranceDict):
    # 因为下面是进行测试，所以不需要计算梯度
    model.eval()
    with torch.no_grad():
        totalEER = 0.0
        numSpeaker = 0
        totalThreshold = 0
        choseSpeakerList = []
        for i in range(Config.getMainConfig("testBatch")):
            choseSpeakerList.append(random.sample(speakerAndUtteranceDict.keys(), 1)[0])
        for i in choseSpeakerList:
            testData, labelList = getTestDataForOneSpeaker(speakerAndUtteranceDict, i)
            testEmbed = model(testData)
            anchorEmbed = testEmbed[0]
            positiveEmbed = testEmbed[1]
            negativeEmbedList = testEmbed[2::]
            similarityList = [getSimilarity(anchorEmbed, positiveEmbed)]
            for negativeEmbed in negativeEmbedList:
                similarityList.append(getSimilarity(anchorEmbed, negativeEmbed))
            # print(similarityList)
            # print("\n")
            bestEER, bestThreshold, bestFRR, bestFAR = getEER(similarityList, labelList)
            totalEER = totalEER + bestEER
            totalThreshold = totalThreshold + bestThreshold
            numSpeaker = numSpeaker + 1
        EER = totalEER/numSpeaker
        threshold = totalThreshold/numSpeaker
        print("测试完毕， EER为: %.3f" % EER)
        print("最合适的阈值是: %.3f" % threshold)
        return EER, threshold