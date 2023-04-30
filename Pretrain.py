import logging
import os

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import Config
import Tools
from DeepSpeakerDataSet import DeepSpeakerDataSet
from Model import SoftmaxDeepSpeakerModel


def testPretrainNet(deepSpeaker, testDataLoader):
    correct = 0
    total = 0

    # 因为下面是进行测试，所以不需要计算梯度
    deepSpeaker.eval()
    with torch.no_grad():
        for _, data in enumerate(testDataLoader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()  # 不要忘了用GPU加速
            outputs = deepSpeaker(inputs)
            # torch.max(a, 1)返回每一行的最大值 torch.max(a，0)返回每一列的最大值
            # torch.max返回的是一个有两个元素的元祖，第一个是每一行（列）最大的数，第二个是每一行（列）最大的数的下标
            # _,代表着忽略第一个，只接受第二个
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            # item()只能对单个元素的tensor使用，将其从向量转化为标量
            # 两个tensor用 == 对比，相同的部分为1，不同的为0，用sum整合，再用item变成标量返回。
            for i in range(labels.__len__()):
                if predicted[i].item() == labels[i].item():
                    correct = correct + 1

    print('测试集正确率:', 100 * correct / total, '%', '\n')

    return 100 * correct / total


def savePretrainCheckpoint(epoch, accuracy, model, better):
    if better:
        print("更好的正确率:" + str(accuracy))

    pretrainModelSaveFileName = "checkpointPretrain_" + str(epoch + 1) + "_" + str(accuracy) + ".pt"
    print("尝试保存预训练检查点:" + pretrainModelSaveFileName)
    pretrainCheckpointFilePath = Config.getPretrainConfig("pretrainCheckpointFilePath")
    pretrainModelSavePath = os.path.join(pretrainCheckpointFilePath, pretrainModelSaveFileName)
    # 如果检查点数量多余10个，就去掉旧的
    Tools.checkFileNumAndDeleteMore(pretrainCheckpointFilePath, 10)

    # 保存模型
    torch.save({
        "epoch": epoch,
        "accuracy": accuracy,
        "modelStateDict": model.state_dict()}, pretrainModelSavePath)
    print("保存完毕")


# 加载检查点
def loadPretrainCheckpoint(model):
    pretrainCheckpointFilePath = Config.getPretrainConfig("pretrainCheckpointFilePath")
    pretrainCheckpointFileName = Tools.getNewestFile(pretrainCheckpointFilePath)
    loadedParas = torch.load(os.path.join(pretrainCheckpointFilePath, pretrainCheckpointFileName))
    model.load_state_dict(loadedParas["modelStateDict"])
    useCheckPointEvaluation = Config.getPretrainConfig("useCheckPointEvaluation")
    if useCheckPointEvaluation:
        accuracy = loadedParas["accuracy"]
    else:
        accuracy = 0
    logging.info(
        "载入预训练检查点模型成功，模型文件为:" + str(pretrainCheckpointFileName) + "，准确率为:" + str(accuracy))
    return model, accuracy


def trainNet():
    # 准备训练环境
    pretrainLossLogFilePath = Config.getPretrainConfig("pretrainLossLogFilePath")
    pretrainAccuracyLogFilePath = Config.getPretrainConfig("pretrainAccuracyLogFilePath")
    # 指定GPU训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备:" + str(device))

    # 初始化训练参数
    bestAccuracy = 0
    useTheCheckpoint = Config.getPretrainConfig("useTheCheckpoint")
    # 加载训练集
    trainSet = DeepSpeakerDataSet(Config.getPretrainConfig("trainSetPath"))
    trainDataLoader = DataLoader(dataset=trainSet, batch_size=Config.getPretrainConfig("trainBatchSize"), shuffle=True)

    # 加载测试集
    testSet = DeepSpeakerDataSet(Config.getPretrainConfig("testSetPath"))
    testDataLoader = DataLoader(dataset=testSet, batch_size=Config.getPretrainConfig("testBatchSize"))

    numSpeaker = trainSet.getNumSpeaker()
    # 加载模型
    deepSpeaker = SoftmaxDeepSpeakerModel(Config.getPretrainConfig("numFrame"), numSpeaker)
    if useTheCheckpoint:
        deepSpeaker, bestAccuracy = loadPretrainCheckpoint(deepSpeaker)
    else:
        open(pretrainLossLogFilePath, 'w').close()
        open(pretrainAccuracyLogFilePath, 'w').close()
    # 将模型迁移到GPU上
    deepSpeaker = deepSpeaker.cuda()
    # 设置损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    # 使用SGD优化
    optimizer = optim.SGD(deepSpeaker.parameters(), lr=Config.getPretrainConfig("learningRate"),
                          momentum=Config.getPretrainConfig("momentum"))

    maxEpoch = Config.getPretrainConfig("maxEpoch")
    for epoch in range(maxEpoch):
        runningLoss = 0.0
        deepSpeaker.train()
        for i, data in enumerate(trainDataLoader):
            inputs, labels = data
            # 清理梯度
            optimizer.zero_grad()
            outputs = deepSpeaker(inputs)
            # 计算损失函数
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss = runningLoss + loss.item()
            # 每十轮batch就输出一次平均loss
            if i % 10 == 9:
                print(
                    '[%d, %5d] loss: %.3f，显存占用: %.2fGB' %
                    (epoch + 1, i + 1, runningLoss / 10,
                     torch.cuda.memory_reserved() / 1073741824))
                Tools.saveInformation(runningLoss / 10, pretrainLossLogFilePath)
                runningLoss = 0.0

        accuracy = testPretrainNet(deepSpeaker, testDataLoader)
        Tools.saveInformation(accuracy, pretrainAccuracyLogFilePath)
        if accuracy > bestAccuracy:
            savePretrainCheckpoint(epoch, accuracy, deepSpeaker, True)
            bestAccuracy = accuracy
        elif accuracy > 0.95:
            savePretrainCheckpoint(epoch, accuracy, deepSpeaker, False)


trainNet()
