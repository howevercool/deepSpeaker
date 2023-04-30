import matplotlib.pyplot as plt
import numpy as np

import Config


def getLogInformation(filePath):
    with open(filePath, 'r') as file:
        informationList = file.read().splitlines()
    result = []
    for i in informationList:
        result.append(float(i))
    return result


def showLogInformation():
    lossLogFilePath = Config.getMainConfig("lossLogFilePath")
    lossList = getLogInformation(lossLogFilePath)
    EERLogFilePath = Config.getMainConfig("EERLogFilePath")
    EERList = getLogInformation(EERLogFilePath)

    # 开始绘图
    fig, area = plt.subplots(1, 2)
    x = np.arange(1, len(lossList) + 1)
    # 绘制loss图像
    area[0].plot(x, lossList)
    area[0].set_title('loss')

    x = np.arange(1, len(EERList) + 1)
    # 绘制EER图像
    area[1].plot(x, EERList)
    area[1].set_title('EER')
    plt.show()


def showPretrainLogInformation():
    lossLogFilePath = Config.getPretrainConfig("pretrainLossLogFilePath")
    lossList = getLogInformation(lossLogFilePath)
    accuracyLogFilePath = Config.getPretrainConfig("pretrainAccuracyLogFilePath")
    accuracyList = getLogInformation(accuracyLogFilePath)

    # 开始绘图
    fig, area = plt.subplots(1, 2)
    x = np.arange(1, len(lossList) + 1)
    # 绘制loss图像
    area[0].plot(x, lossList)
    area[0].set_title('loss')

    x = np.arange(1, len(accuracyList) + 1)
    # 绘制EER图像
    area[1].plot(x, accuracyList)
    area[1].set_title('accuracy')
    plt.show()


def showFitness():
    maxFitnessLogFilePath = Config.getDEConfig("maxFitnessLogFilePath")
    maxFitnessList = getLogInformation(maxFitnessLogFilePath)
    averageFitnessLogFilePath = Config.getDEConfig("averageFitnessLogFilePath")
    averageFitnessList = getLogInformation(averageFitnessLogFilePath)

    # 开始绘图
    fig, area = plt.subplots(1, 2)
    x = np.arange(1, len(maxFitnessList) + 1)
    # 绘制loss图像
    area[0].plot(x, maxFitnessList)
    area[0].set_title('max fitness')

    x = np.arange(1, len(averageFitnessList) + 1)
    # 绘制EER图像
    area[1].plot(x, averageFitnessList)
    area[1].set_title('average fitness')
    plt.show()


showPretrainLogInformation()