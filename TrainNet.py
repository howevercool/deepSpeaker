import logging
import os

import torch
from torch import optim

import Config
import Tools
from Model import DeepSpeakerModel
from TestNet import testNet
from TripletLoss import getSpeakerAndUtteranceDict, TripletLoss, Triplet


def saveCheckpoint(epoch, EER, threshold, model, better):
    if better:
        print("更好的EER: %.3f" % EER)
    modelSaveFileName = "checkpoint_" + str(epoch + 1) + "_" + str(EER) + ".pt"
    print("尝试保存检查点:" + modelSaveFileName)
    checkpointFilePath = Config.getMainConfig("checkpointFilePath")
    modelSavePath = os.path.join(checkpointFilePath, modelSaveFileName)

    Tools.checkFileNumAndDeleteMore(checkpointFilePath, 10)

    # 保存模型
    torch.save({
        "epoch": epoch,
        "EER": EER,
        "threshold": threshold,
        "modelStateDict": model.state_dict()}, modelSavePath)
    print("保存完毕")


# 加载检查点
def loadCheckpoint(model):
    checkpointFilePath = Config.getMainConfig("checkpointFilePath")
    checkpointFileName = Tools.getNewestFile(checkpointFilePath)
    loadedParas = torch.load(os.path.join(checkpointFilePath, checkpointFileName))

    model.load_state_dict(loadedParas["modelStateDict"])
    useCheckPointEvaluation = Config.getMainConfig("useCheckPointEvaluation")
    if useCheckPointEvaluation:
        EER = loadedParas["EER"]
    else:
        EER = 1
    logging.info("载入检查点模型成功，模型文件为:" + str(checkpointFileName) + "，EER为:" + str(EER))
    return model, EER


def loadPretrainModel(model):
    pretrainModelPath = Config.getMainConfig("pretrainModelPath")
    loadedParas = torch.load(pretrainModelPath)
    model.load_state_dict(loadedParas["modelStateDict"], strict=False)
    return model


def trainNet():
    # 初始化时间测量器
    timeMeasure = Tools.TimeMeasure()
    lossLogFilePath = Config.getMainConfig("lossLogFilePath")
    open(lossLogFilePath, 'w').close()
    EERLogFilePath = Config.getMainConfig("EERLogFilePath")
    open(EERLogFilePath, 'w').close()

    bestEER = 1.0
    # 指定GPU训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备:" + str(device))

    # 加载模型
    deepSpeaker = DeepSpeakerModel()
    # 设置损失函数
    criterion = TripletLoss(Config.getMainConfig("tripletLossMargin")).cuda()
    # 使用SGD优化
    optimizer = optim.SGD(deepSpeaker.parameters(), lr=Config.getMainConfig("learningRate"),
                          momentum=Config.getMainConfig("momentum"))

    usePretrainedModel = Config.getMainConfig("usePretrainedModel")
    if usePretrainedModel:
        print("尝试加载预训练模型")
        deepSpeaker = loadPretrainModel(deepSpeaker)
    else:
        useTheCheckpoint = Config.getMainConfig("useTheCheckpoint")
        if useTheCheckpoint:
            deepSpeaker, bestEER = loadCheckpoint(deepSpeaker)
    # 将模型迁移到GPU上
    deepSpeaker = deepSpeaker.cuda()
    trainSetSpeakAndUtteranceDict = getSpeakerAndUtteranceDict(Config.getMainConfig("trainSetPath"))
    testSetSpeakAndUtteranceDict = getSpeakerAndUtteranceDict(Config.getMainConfig("testSetPath"))

    maxEpoch = Config.getMainConfig("maxEpoch")
    triplet = Triplet(trainSetSpeakAndUtteranceDict, Config.getMainConfig("tripletLossMargin"))
    testPerBatch = Config.getMainConfig("testPerBatch")
    # deepSpeaker.eval()
    # print("网络初始测试")
    # timeMeasure.start()
    # EER, threshold = testNet(deepSpeaker, testSetSpeakAndUtteranceDict)
    # timeMeasure.printTimeWithMessage("测试用时")

    for epoch in range(maxEpoch):
        runningLoss = 0.0
        deepSpeaker.train()
        timeMeasure.start()
        for i in range(Config.getMainConfig("testPerBatch")):
            optimizer.zero_grad()
            filterBankData = triplet.getBatch(deepSpeaker)
            embed = deepSpeaker(filterBankData)
            loss = criterion(embed, triplet.batchSize)
            loss.backward()
            optimizer.step()
            runningLoss = runningLoss + loss.item()
            # 每十轮batch就输出一次平均loss
            if i % 10 == 9:
                spendTime = timeMeasure.getSpendTime()
                print(
                    '[%d, %5d] loss: %.3f，显存占用: %.2fGB，每轮batch耗时: %.2fs' %
                    (epoch + 1, i + 1, runningLoss / 10,
                     torch.cuda.memory_reserved() / 1073741824,
                     spendTime / 10))
                Tools.saveInformation(runningLoss / 10, lossLogFilePath)
                runningLoss = 0.0
            if i % testPerBatch == testPerBatch - 1:
                print("开始测试网络")
                timeMeasure.start()
                EER, threshold = testNet(deepSpeaker, testSetSpeakAndUtteranceDict)
                timeMeasure.printTimeWithMessage("测试用时")
                Tools.saveInformation(EER, EERLogFilePath)
                deepSpeaker.train()
                if EER < bestEER:
                    saveCheckpoint(epoch, EER, threshold, deepSpeaker, True)
                    bestEER = EER
                elif EER < 0.05:
                    saveCheckpoint(epoch, EER, threshold, deepSpeaker, False)


if __name__ == '__main__':
    trainNet()
