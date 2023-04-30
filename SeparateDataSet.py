import math
import os
import shutil
from random import *

workSpacePath = r"D:\pythonProject\DeepSpeaker\dataSet\filterBankDataSet\AllData"

trainSetPath = r"D:\pythonProject\DeepSpeaker\dataSet\filterBankDataSet\TrainSet"

testSetPath = r"D:\pythonProject\DeepSpeaker\dataSet\filterBankDataSet\TestSet"

separateScale = 0.2


# 从训练集中随机选出数据作为测试集
def SeparateData(workSpacePathNow, trainSetPathNow, testSetPathNow):
    global separateScale
    if not os.path.exists(testSetPathNow):
        os.makedirs(testSetPathNow)
    if not os.path.exists(trainSetPathNow):
        os.makedirs(trainSetPathNow)

    files = os.listdir(workSpacePathNow)
    if os.path.isfile(workSpacePathNow + '\\' + files[0]):
        if len(files) < 2:
            print("数据过少，无法划分数据集")
            exit(-1)
        # 计算需要放到测试集的数量
        testDataNumber = math.ceil(len(files) * separateScale)
        # 控制测试集和训练集都至少有一个数据
        if testDataNumber == len(files):
            testDataNumber = testDataNumber - 1
        elif testDataNumber == 0:
            testDataNumber = 1

        # 随机选取要移动到测试集的文件
        testDataList = sample(files, testDataNumber)

        # 复制到测试集
        for i in testDataList:
            shutil.copyfile(os.path.join(workSpacePathNow, i), os.path.join(testSetPathNow, i))

        # 复制到训练集
        files = list(set(files).difference(set(testDataList)))
        for file in files:
            shutil.copyfile(os.path.join(workSpacePathNow, file), os.path.join(trainSetPathNow, file))
    else:
        for file in files:
            SeparateData(os.path.join(workSpacePathNow, file), os.path.join(trainSetPathNow, file),
                         os.path.join(testSetPathNow, file))


def main():
    SeparateData(workSpacePath, trainSetPath, testSetPath)


if __name__ == '__main__':
    main()
