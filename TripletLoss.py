import random

import torch
from torch import nn

import Config
import Tools
from Tools import selectFiles, getFilterBankData


def getSimilarityMetrix(embedMetrix):
    return torch.mm(embedMetrix, embedMetrix.t())


# def getBatch(speakerAndUtteranceDict, model):
#     batchSize = Config.getMainConfig("batchSize")
#     # 本次batch使用的语句
#     speakerAndUtteranceThisBatchUseDict = {}
#     utteranceList = []
#     anchorList = []
#     positiveList = []
#     negativeList = []
#
#     for speaker in speakerAndUtteranceDict:
#         speakerAndUtteranceThisBatchUseDict[speaker] = random.sample(speakerAndUtteranceDict[speaker],
#                                                                      numUtteranceOfOneSpeaker)
#         sizeOfUtteranceList = len(utteranceList)
#         utteranceList.extend(speakerAndUtteranceThisBatchUseDict[speaker])
#         # 每个说话人产生numUtteranceOfOneSpeaker * (numUtteranceOfOneSpeaker - 1)个三元组
#         for anchorIndex in range(numUtteranceOfOneSpeaker):
#             numPositive = numUtteranceOfOneSpeaker - 1
#             for _ in range(numPositive):
#                 anchorList.append([speakerAndUtteranceThisBatchUseDict[speaker][anchorIndex],
#                                    sizeOfUtteranceList + anchorIndex])
#             positiveIndex = 0
#             for i in range(numUtteranceOfOneSpeaker):
#                 while positiveIndex < numUtteranceOfOneSpeaker:
#                     if positiveIndex == anchorIndex:
#                         positiveIndex += 1
#                         continue
#                     positiveList.append([speakerAndUtteranceThisBatchUseDict[speaker][positiveIndex],
#                                          sizeOfUtteranceList + positiveIndex])
#                     positiveIndex += 1
#                     break
#
#     # 接下来搞定negativeList
#     utteranceFilterBankDataList = []
#     numUtterance = len(utteranceList)
#     for i in range(numUtterance):
#         filterBankDataPath = utteranceList[i]
#         filterBankData = getFilterBankData(filterBankDataPath)
#         utteranceFilterBankDataList.append(filterBankData)
#     utteranceFilterBankDataList = torch.stack(utteranceFilterBankDataList).cuda()
#
#     utteranceEmbed = model(utteranceFilterBankDataList)
#
#     similarityMetrix = getSimilarityMetrix(utteranceEmbed)
#
#     # 开始为每个三元组挑选negative
#     for i in anchorList:
#         anchorIndex = i[1]
#         similarityToAnchor = similarityMetrix[anchorIndex]
#         for j in range(numUtteranceOfOneSpeaker):
#             similarityToAnchor[int(anchorIndex / numUtteranceOfOneSpeaker + j)] = -1
#         negativeIndex = torch.argmax(similarityToAnchor)
#         negativeList.append([utteranceList[negativeIndex], negativeIndex.item()])
#
#     # 成功get三元祖，下面是获取数据并返回
#     anchorFilterBankDataMetrix = []
#     positiveFilterBankDataMetrix = []
#     negativeFilterBankDataMetrix = []
#     filterBankDataMetrix = []
#
#     for i in anchorList:
#         filterBankDataMetrix.append(utteranceFilterBankDataList[i[1]])
#
#     for i in positiveList:
#         filterBankDataMetrix.append(utteranceFilterBankDataList[i[1]])
#
#     for i in negativeList:
#         filterBankDataMetrix.append(utteranceFilterBankDataList[i[1]])
#
#     return filterBankDataMetrix


def getHardestTripletBatch():
    pass


# 获取一个dict，key是说话人id，value是这个说话人说的话的文件路径
def getSpeakerAndUtteranceDict(dataSetPath):
    speakerAndUtteranceDict = {}
    # 获取数据列表
    trainSetFileList = selectFiles(dataSetPath + "\\**\\*.pt", True)
    for i in range(len(trainSetFileList)):
        fileName = trainSetFileList[i].split('\\')[-1]
        speaker = fileName.split('-')[0]
        if speaker not in speakerAndUtteranceDict:
            speakerAndUtteranceDict[speaker] = []
        speakerAndUtteranceDict[speaker].append(trainSetFileList[i])
    return speakerAndUtteranceDict


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        # 设置margin值
        self.margin = margin

    # # 耗时低，效率还可以
    # def forward(self, anchorEmbedMetrix, positiveEmbedMetrix, negativeEmbedMetrix):
    #     similarityAP = torch.einsum("ij,ij->i", [anchorEmbedMetrix, positiveEmbedMetrix])
    #     similarityAN = torch.einsum("ij,ij->i", [anchorEmbedMetrix, negativeEmbedMetrix])
    #     tripletLossTensor = torch.clamp(self.margin + similarityAN - similarityAP, min=0.0)
    #     loss = torch.mean(tripletLossTensor)
    #     return loss

    def forward(self, embed, numTriplet):
        anchorEmbed = embed[:2 * numTriplet:2]
        positiveEmbed = embed[1 : 2*numTriplet + 1:2]
        negativeEmbed = embed[2*numTriplet::]
        similarityAP = torch.einsum("ij,ij->i", [anchorEmbed, positiveEmbed])
        similarityAN = torch.einsum("ij,ij->i", [anchorEmbed, negativeEmbed])
        tripletLossTensor = torch.clamp(self.margin + similarityAN - similarityAP, min=0.0)
        loss = torch.mean(tripletLossTensor)
        return loss


class Triplet:
    def __init__(self, speakerAndUtteranceDict, margin):
        self.historySize = Config.getMainConfig("historySize")
        # 指向最久的历史数据，用来指示下一次替换掉哪一块
        self.oldestIndex = 0
        self.historyFilterBankDataList = []
        self.historyEmbed = []
        self.historyLabelList = []
        # 每个batch，返回多少个三元组
        self.batchSize = Config.getMainConfig("batchSize")
        self.speakerAndUtteranceDict = speakerAndUtteranceDict
        # 说话人id的列表，实际上是一个dict_keyss
        self.speakerList = speakerAndUtteranceDict.keys()
        self.margin = margin

    # 应该提供filterBankData
    def getBatch(self, model):
        model.eval()
        with torch.no_grad():
            utteranceList = []
            speakerList = []
            for i in range(self.batchSize):
                anchorSpeaker = random.sample(self.speakerList, 1)[0]
                anchorPositivePair = random.sample(self.speakerAndUtteranceDict[anchorSpeaker], 2)
                utteranceList.extend(anchorPositivePair)
                speakerList.append(anchorSpeaker)
                speakerList.append(anchorSpeaker)

            # 这个批次的filterBankData
            filterBankData = []
            for utterancePath in utteranceList:
                filterBankData.append(torch.unsqueeze(Tools.getFilterBankData(utterancePath), 0))
            filterBankData = torch.stack(filterBankData)

            embed = model(filterBankData)

            # 如果未达到历史列表上限，就可以直接往里面加数据
            if len(self.historyFilterBankDataList) < self.historySize * self.batchSize * 2:
                self.historyFilterBankDataList.extend(filterBankData)
                self.historyLabelList.extend(speakerList)
                self.historyEmbed.extend(embed)
            # 否则就替换掉最旧的数据
            else:
                self.historyFilterBankDataList[self.oldestIndex * self.batchSize * 2:
                                               (self.oldestIndex + 1) * self.batchSize * 2] = filterBankData
                self.historyLabelList[self.oldestIndex * self.batchSize * 2:
                                      (self.oldestIndex + 1) * self.batchSize * 2] = speakerList
                self.historyEmbed[self.oldestIndex * self.batchSize * 2:
                                      (self.oldestIndex + 1) * self.batchSize * 2] = embed

            anchorEmbed = embed[::2]
            positiveEmbed = embed[1::2]
            similarityAP = torch.einsum("ij,ij->i", [anchorEmbed, positiveEmbed])

            # 开始获取negative
            negativeEmbedIndex = []
            negativeFilterBankData = []
            historyEmbedTorch = torch.stack(self.historyEmbed)
            for i in range(len(anchorEmbed)):
                maxSimilarity = -2
                bestNegativeIndex = 0
                # 计算相似度并选出满足条件的negative
                similarityList = torch.matmul(historyEmbedTorch, anchorEmbed[i].t()).squeeze()
                for k in range(len(similarityList)):
                    if self.historyLabelList[k] == speakerList[2 * i]:
                        continue
                    else:
                        tripletLoss = similarityList[k] - similarityAP[i] + self.margin
                        if tripletLoss > 0:
                            bestNegativeIndex = k
                            break
                        if similarityList[k] > maxSimilarity:
                            maxSimilarity = similarityList[k]
                            bestNegativeIndex = k
                negativeEmbedIndex.append(bestNegativeIndex)

            for i in range(len(negativeEmbedIndex)):
                negativeFilterBankData.append(self.historyFilterBankDataList[negativeEmbedIndex[i]])
            negativeFilterBankData = torch.stack(negativeFilterBankData)

            filterBankData = torch.cat((filterBankData, negativeFilterBankData), dim=0)
            # 收尾工作
            self.oldestIndex = (self.oldestIndex + 1) % self.historySize
            model.train()
            return filterBankData

    # def getBatchOld(self, model):
    #     # 说话人id的列表，实际上是一个dict_keys
    #     utteranceList = []
    #     anchorList = []
    #     positiveList = []
    #     negativeList = []
    #     for i in range(self.batchSize):
    #         anchorSpeaker = random.sample(self.speakerList, 1)[0]
    #         anchorPositivePair = random.sample(self.speakerAndUtteranceDict[anchorSpeaker], 2)
    #         anchorList.append((anchorPositivePair[0], anchorSpeaker))
    #         positiveList.append((anchorPositivePair[1], anchorSpeaker))
    #         utteranceList.append((anchorPositivePair[0], anchorSpeaker))
    #         utteranceList.append((anchorPositivePair[1], anchorSpeaker))
    #
    #     # 如果未达到历史列表上限，就可以直接往里面加数据
    #     if len(self.historyFilterBankDataList) < self.historySize * self.batchSize * 2:
    #         # 获取本轮使用的filterBankData
    #         filterBankDataList = []
    #         for utterance in utteranceList:
    #             filterBankDataList.append(Tools.getFilterBankData(utterance[0]))
    #
    #     # 获取本轮使用的embed
    #
    #     if len(self.historyUtteranceList) < self.historySize * self.batchSize * 2:
    #         self.historyUtteranceList.extend(utteranceList)
    #         self.historyFilterBankDataList.extend(filterBankDataList)
    #     else:
    #         self.historyUtteranceList[self.oldestIndex * self.batchSize * 2:
    #                                   (self.oldestIndex + 1) * self.batchSize * 2] = utteranceList
    #
    #     historyFilterBankDataList = []
    #     # 获取历史数据的filterBankData
    #     for i in range(len(self.historyUtteranceList)):
    #         historyFilterBankDataList.append(Tools.getFilterBankData(self.historyUtteranceList[i][0]))
    #     historyFilterBankDataMetrix = torch.stack(historyFilterBankDataList).cuda()
    #     historyEmbed = model(historyFilterBankDataMetrix)
    #
    #     anchorEmbed = historyEmbed[self.oldestIndex * self.batchSize * 2:(self.oldestIndex + 1) * self.batchSize * 2:2]
    #     positiveEmbed = historyEmbed[self.oldestIndex * self.batchSize * 2 + 1:
    #                                  (self.oldestIndex + 1) * self.batchSize * 2 + 1:2]
    #     similarityAP = torch.einsum("ij,ij->i", [anchorEmbed, positiveEmbed])
    #
    #     # 挑选negative
    #     negativeEmbed = []
    #     for i in range(len(anchorEmbed)):
    #         similarityList = torch.matmul(historyEmbed, anchorEmbed[i].t()).squeeze()
    #         largestSimilarity = 0
    #         negativeUtteranceIndex = 0
    #         getSemiHardNegative = False
    #         for j in range(len(similarityList)):
    #             if self.historyUtteranceList[j][1] == anchorList[i][1]:
    #                 continue
    #             else:
    #                 tripletLoss = similarityList[j] - similarityAP[i] + self.margin
    #                 if tripletLoss > 0:
    #                     negativeEmbed.append(historyEmbed[j])
    #                     getSemiHardNegative = True
    #                     break
    #                 if similarityList[j] > largestSimilarity:
    #                     largestSimilarity = similarityList[j]
    #                     negativeUtteranceIndex = j
    #         if not getSemiHardNegative:
    #             negativeEmbed.append(historyEmbed[negativeUtteranceIndex])
    #     negativeEmbed = torch.stack(negativeEmbed)
    #     similarityAN = torch.einsum("ij,ij->i", [anchorEmbed, negativeEmbed])
    #     self.oldestIndex = (self.oldestIndex + 1) % self.historySize
    #     return similarityAP, similarityAN
