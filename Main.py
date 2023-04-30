import logging

import Config
import SeparateDataSet
from ProcessData import processData
from TrainNet import trainNet
from TripletLoss import getSpeakerAndUtteranceDict, getBatch


def main():
    # 全局配置
    logging.basicConfig(level=logging.INFO)
    # 开始加载特征文件
    recreateFeatureFile = Config.getMainConfig("recreateFeatureFile")
    if recreateFeatureFile:
        processData()
    separateDataSet = Config.getMainConfig("separateDataSet")
    if separateDataSet:
        SeparateDataSet.main()
    trainNet()


if __name__ == '__main__':
    main()
