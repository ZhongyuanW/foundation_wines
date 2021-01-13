# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 下午2:04
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : wineDataset.py
# @Software: PyCharm

from torch.utils.data import Dataset as dataset
import json
import torch
from datetime import datetime

class WineDataset(dataset):
    def __init__(self, anno_file="datasets/wines.json"):
        self.__annotations = self.__load_annotations(anno_file)


    def __load_annotations(self, anno_file):
        with open(anno_file, "r") as f:
            labels = []
            datas = json.load(f)

            annos = datas["Data"]["LSJZList"][::-1]
            self.base_date = datetime.strptime(annos[0]["FSRQ"],"%Y-%m-%d")
            for i in annos:
                if i["FSRQ"] and i["LJJZ"]:
                    date = (datetime.strptime(i["FSRQ"],"%Y-%m-%d")-self.base_date).days
                    labels.append([date, float(i["LJJZ"])])
            return labels

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        date, data = self.__annotations[item]
        return torch.FloatTensor([[[date]]]), torch.FloatTensor([[[data]]])
        # return torch.FloatTensor([date]), torch.FloatTensor([data])

if __name__ == "__main__":
    import torch.utils.data.dataloader as Dataloader
    dataset = WineDataset("wines.json")

    dataloader = Dataloader.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)

    for i, (x,y) in enumerate(dataloader):
        print(x.shape, y.shape)

