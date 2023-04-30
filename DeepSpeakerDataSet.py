import os

import numpy as np
import torch
from torch.utils.data import Dataset

import Config
import Tools
from Tools import cutAudio

labelList = []


class DeepSpeakerDataSet(Dataset):
    def __init__(self, filterBankDataSetPath):
        # 用来保存数据，其中的每个元素都是一个二元组，第一个元素是特征数据，第二个元素是标签
        self.dataAndLabelSet = []
        self.labelList = []
        self.labelList = os.listdir(filterBankDataSetPath)
        for i in self.labelList:
            workSpace = filterBankDataSetPath + "\\" + i
            for j in os.listdir(workSpace):
                self.dataAndLabelSet.append((workSpace + "\\" + j, torch.as_tensor(self.labelList.index(i)).cuda()))

    def __getitem__(self, index):
        filterBankDataPath = self.dataAndLabelSet[index][0]
        filterBankData = Tools.getFilterBankData(filterBankDataPath)
        filterBankData = torch.unsqueeze(filterBankData, 0)
        data = (filterBankData, self.dataAndLabelSet[index][1])
        return data

    def __len__(self):
        return len(self.dataAndLabelSet)

    def getNumSpeaker(self):
        return len(self.labelList)
