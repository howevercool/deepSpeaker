# 从语料库中找到和目标说话人最相近的50个句子
import heapq
import math
import os
import random

import librosa
import numpy as np
import torch
from torch import float32

import Config
import ProcessData
import Tools
from Model import DeepSpeakerModel
from Tools import selectFiles, getSimilarity

# 说话人识别模型
model = DeepSpeakerModel()
# 验证说话人时，使用的相似度阈值
threshold = 0
# 音频长度
audioLength = 0
# 音频每个切片的长度
audioSliceLength = Config.getDEConfig("audioSliceLength")
#  音频可以切分为多少个片
numSlice = 0
sampleRate = Config.getMainConfig("sampleRate")


# 候选点类
class CandidatePoint:
    def __init__(self, basePosition, positionOffset, turbulenceValue):
        self.basePosition = basePosition
        self.positionOffset = positionOffset
        self.turbulenceValue = turbulenceValue

    # 使用候选点的攻击信息，生成用于攻击的对抗样本
    def generateAttackData(self, audio):
        attackData = audio.copy()
        attackData[self.basePosition * audioSliceLength + self.positionOffset] += self.turbulenceValue
        return attackData

    # 将对抗样本保存下来
    def saveAsAttackData(self, audio, path):
        audio[self.basePosition * audioSliceLength + self.positionOffset] += self.turbulenceValue
        print("这里还未完工")
        exit(-1)

    def clamp(self):
        # 进行边界处理
        if self.basePosition < 0:
            self.basePosition = 0
        if self.basePosition > numSlice - 1:
            self.basePosition = numSlice - 1

        if self.positionOffset < 0:
            self.positionOffset = 0
        if self.positionOffset > audioSliceLength - 1:
            self.positionOffset = audioSliceLength - 1

        if self.basePosition == numSlice - 1 and \
                self.basePosition * audioSliceLength + self.positionOffset > audioLength - 1:
            self.positionOffset = audioLength - 1 - self.basePosition * audioSliceLength


# 这个函数是用来获取所有待攻击数据的嵌入的，方便之后可以和说话人嵌入进行相似度对比
def processAttackDataSet(attackDataSetPath, attackEmbedSetPath):
    global model
    utterancePathList = selectFiles(attackDataSetPath + "\\**\\*.flac", True)
    for utterancePath in utterancePathList:
        filterBankData = ProcessData.processOneUtterance(utterancePath)
        filterBankData = filterBankData.reshape(1, -1, 64)
        filterBankData = Tools.padFilterBankData(filterBankData)
        filterBankData = torch.as_tensor(filterBankData, dtype=float32).cuda()
        filterBankData = torch.unsqueeze(filterBankData, 0)
        embed = model(filterBankData)
        embed = embed.squeeze()
        embedSavePath = os.path.join(attackEmbedSetPath, utterancePath.split("\\")[-1].split(".")[-2] + ".pt")
        torch.save(embed, embedSavePath)


# 获取音频和说话人嵌入的相似度
def getSimilarityWithSpeakerEmbed(audio, speakerEmbed):
    global model
    audio = ProcessData.VAD(audio.flatten(), sampleRate)
    filterBankData = ProcessData.getFilterBankDataFromAudio(audio)
    filterBankData = Tools.padFilterBankData(filterBankData)
    filterBankData = torch.as_tensor(filterBankData, dtype=float32).cuda()
    filterBankData = torch.unsqueeze(filterBankData, 0)
    embed = model(filterBankData)
    embed = torch.squeeze(embed)
    similarity = getSimilarity(speakerEmbed, embed)
    return similarity


def get50SimilarityFilePathList(speakerEmbed, attackEmbedSetPath):
    global model
    similarityList = []
    embedSetPath = selectFiles(attackEmbedSetPath + "\\**\\*.pt", True)

    # 以下代码是用来辨别是否有同属于一个说话人的语句。但是因为使用的数据局中，注册集和攻击集是分开的，所以不需要了。
    # for i in allFilterBankDataFilePath:
    #     if not i.split("\\")[-3] == speakerId:
    #         candidateFilterBankDataFilePath.append(i)
    for embedPath in embedSetPath:
        embed = torch.load(embedPath)
        similarityList.append(Tools.getSimilarity(speakerEmbed, embed))
    similarityList = np.array(similarityList)
    maxSaps = heapq.nlargest(50, similarityList)
    # 找到最不像锚点数据的正向数据的下标
    maxSimilarityUtterancePathList = []
    attackDataSetPath = Config.getDEConfig("attackDataSetPath")
    # 通过embed文件名来获取音频的路径
    for i in maxSaps:
        embedPath = embedSetPath[np.argwhere(similarityList == i).flatten()[0]]
        fileName = embedPath.split("\\")[-1].split(".")[0] + ".flac"
        filePath = os.path.join(attackDataSetPath,
                                fileName.split("-")[0] + "\\" + fileName.split("-")[1] + "\\" + fileName)
        maxSimilarityUtterancePathList.append(filePath)
    return maxSimilarityUtterancePathList


# 初始化种群，随机初始化
def getInitPopulation():
    global audioLength, audioSliceLength, numSlice
    population = []
    initPopulationSize = Config.getDEConfig("initPopulationSize")
    for i in range(initPopulationSize):
        # 每个filterBankData都是160*64的数据
        basePosition = random.randint(1, numSlice) - 1
        positionOffset = random.randint(1, audioSliceLength) - 1
        turbulenceValue = random.uniform(-100, 100)
        candidatePoint = CandidatePoint(basePosition, positionOffset, turbulenceValue)
        candidatePoint.clamp()
        population.append(candidatePoint)
    return population


# 适应度函数，将候选点注入到数据中，生成攻击数据，计算与目标说话人的相似度
def getFitness(candidate, audio, speakerEmbed):
    global model, threshold
    attackData = candidate.generateAttackData(audio)
    similarity = getSimilarityWithSpeakerEmbed(attackData, speakerEmbed)
    return similarity


# 轮盘赌算法
def RWS(fitnessList, fitnessSum, populationSize):
    randomSelect = random.uniform(0, fitnessSum)
    fitnessTempSum = 0
    for i in range(populationSize):
        tempFitness = fitnessList[i]
        fitnessTempSum = fitnessTempSum + tempFitness
        if randomSelect <= fitnessTempSum:
            return i


# 使用两个随机个体，进行差分变异，并添加到最优解上，生成一个子代个体
def getMutateCandidatePoint(bestCandidatePoint, dad, mom):
    global numSlice, audioSliceLength, audioLength
    scaleFactor = Config.getDEConfig("scaleFactor")
    basePosition = dad.basePosition - mom.basePosition
    positionOffset = dad.positionOffset - mom.positionOffset

    basePosition = int(scaleFactor * basePosition)
    positionOffset = int(scaleFactor * positionOffset)

    turbulenceValue = dad.turbulenceValue - mom.turbulenceValue
    turbulenceValue = scaleFactor * turbulenceValue

    # 获得了差分后的数值，接下来添加到最优解上，生成子代候选点
    basePosition = bestCandidatePoint.basePosition + basePosition
    positionOffset = bestCandidatePoint.positionOffset + positionOffset
    turbulenceValue = bestCandidatePoint.turbulenceValue + turbulenceValue

    candidatePoint = CandidatePoint(basePosition, positionOffset, turbulenceValue)
    candidatePoint.clamp()
    return candidatePoint


# 获取交叉概率
def getCrossPr(fitness, minFitness, maxFitness, averageFitness):
    if fitness <= averageFitness:
        return 0.1
    else:
        # 中间变量，没啥意义
        var = (fitness - minFitness) / (maxFitness - minFitness)
        crossPr = 0.1 + 0.5 * var
        return crossPr


def attack(speakerEmbed, audio):
    global audioLength, audioSliceLength, numSlice
    audioLength = len(audio)
    numSlice = math.ceil(audioLength / audioSliceLength)
    population = getInitPopulation()
    updateFlagList = []
    fitnessList = []
    for i in range(len(population)):
        updateFlagList.append(True)
        fitnessList.append(0)
    populationSize = Config.getDEConfig("initPopulationSize")
    maxIteration = Config.getDEConfig("maxIteration")
    bestSimilarity = -2
    noRaiseNum = 0
    print("开始迭代")
    for iteration in range(maxIteration):
        print("世代:", iteration + 1)
        newPopulation = []
        # 计算整个种群的得分
        maxFitness = -2
        minFitness = 2
        bestCandidate = 0
        fitnessSum = 0
        for i in range(populationSize):
            fitness = 0
            if updateFlagList[i]:
                fitness = getFitness(population[i], audio, speakerEmbed)
                fitnessList[i] = fitness
            else:
                fitness = fitnessList[i]
            fitnessSum += fitness
            if fitness > maxFitness:
                maxFitness = fitness
                bestCandidate = population[i]
            if fitness < minFitness:
                minFitness = fitness
        Tools.saveInformation(maxFitness, Config.getDEConfig("maxFitnessLogFilePath"))
        if maxFitness > bestSimilarity:
            bestSimilarity = maxFitness
            noRaiseNum = 0
            print("最优解有提升")
        else:
            noRaiseNum += 1
            print("最优解无提升次数: ", noRaiseNum)
            if noRaiseNum > 99:
                return False, None, bestSimilarity
        if maxFitness > threshold:
            return True, bestCandidate, bestSimilarity
        averageFitness = fitnessSum / populationSize
        Tools.saveInformation(averageFitness, Config.getDEConfig("averageFitnessLogFilePath"))
        print("最优解相似度:", maxFitness)
        print("平均相似度:", averageFitness)
        # 开始对每个候选点进行变异和交叉
        for i in range(populationSize):
            # 随机选出两个个体，进行差分
            dad = RWS(fitnessList, fitnessSum, populationSize)
            mom = RWS(fitnessList, fitnessSum, populationSize)
            mutatedCandidatePoint = getMutateCandidatePoint(bestCandidate, population[dad], population[mom])
            # 突变结束，接下来进行交叉
            # 首先获得交叉概率
            crossPr = getCrossPr(fitnessList[i], minFitness, maxFitness, averageFitness)
            # 接下来确定哪一个维度一定会交叉
            # TODO 这里应该是一定会交叉还是一定不会交叉？先不写这个部分了
            parentCandidatePoint = [population[i].basePosition, population[i].positionOffset,
                                    population[i].turbulenceValue]
            childCandidatePoint = [mutatedCandidatePoint.basePosition, mutatedCandidatePoint.positionOffset,
                                   mutatedCandidatePoint.turbulenceValue]
            newCandidatePoint = []
            for dim in range(len(parentCandidatePoint)):
                changeRandom = random.random()
                if changeRandom < crossPr:
                    newCandidatePoint.append(childCandidatePoint[dim])
                else:
                    newCandidatePoint.append(parentCandidatePoint[dim])
            newCandidatePoint = CandidatePoint(newCandidatePoint[0], newCandidatePoint[1], newCandidatePoint[2])
            newCandidatePoint.clamp()
            newFitness = getFitness(newCandidatePoint, audio, speakerEmbed)
            if newFitness > fitnessList[i]:
                newPopulation.append(newCandidatePoint)
                updateFlagList[i] = True
            else:
                newPopulation.append(population[i])
                updateFlagList[i] = False
        population = newPopulation
    return False, None, bestSimilarity


def main():
    global model, threshold
    open(Config.getDEConfig("maxFitnessLogFilePath"), 'w').close()
    open(Config.getDEConfig("averageFitnessLogFilePath"), 'w').close()
    # 加载模型
    model, threshold = Tools.loadModelWithEval()
    speakerId = '19'
    threshold = 0.8
    print("阈值为:", threshold)
    speakerEmbed = Tools.getSpeakerEmbed(speakerId)
    # 寻找最相似的50个语句
    recreateSimilarityFileList = Config.getDEConfig("recreateSimilarityFileList")
    # 存放这50个文件地址的文件地址。
    similarityFilePath = Config.getDEConfig("similarityFilePath")
    if recreateSimilarityFileList:
        attackEmbedSetPath = Config.getDEConfig("attackEmbedSetPath")
        similarityFilePathList = get50SimilarityFilePathList(speakerEmbed, attackEmbedSetPath)
        similarityFile = open(similarityFilePath, mode='w')
        for i in similarityFilePathList:
            similarityFile.write(i)
            similarityFile.write("\n")
        similarityFile.close()
    else:
        similarityFile = open(similarityFilePath, mode='r')
        similarityFilePathList = similarityFile.read().splitlines()

    sampleRate = Config.getMainConfig("sampleRate")
    # 获取到了文件列表，开始循环使用这些文件进行攻击
    for filePath in similarityFilePathList:
        # 获取语音数据
        audio = librosa.load(filePath, sr=sampleRate, mono=True)[0]
        # 开始攻击
        attackResult, candidatePoint, bestSimilarity = attack(speakerEmbed, audio)
        if not attackResult:
            continue
        else:
            print("攻击成功，对抗样本的相似度:", bestSimilarity)
            candidatePoint.saveAsAttackData(audio, Config.getDEConfig("candidateSavePath"))
            break


with torch.no_grad():
    # # 加载模型
    # model, threshold = Tools.loadModelWithEval()
    # attackDataSetPath = Config.getDEConfig("attackDataSetPath")
    # attackEmbedSetPath = Config.getDEConfig("attackEmbedSetPath")
    # processAttackDataSet(attackDataSetPath, attackEmbedSetPath)
    main()
