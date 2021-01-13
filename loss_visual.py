# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 下午3:04
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : loss_visual.py
# @Software: PyCharm

import os
import matplotlib.pyplot as plt
import json


vis_to_lossname = ["lr", "loss"]
def visual_loss_from_file(file_path, loss_index, start=0, smooth=True):
    if not os.path.exists(file_path):
        assert "not find {0}!".format(file_path)
    plt.clf()
    with open(file_path, "r") as f:
        datas = [list(map(str.strip,i.split(", "))) for i in filter(lambda x:"epoch" in x,f.readlines())]
    loss = [float(i[loss_index + 2].split(" ")[-1]) for i in datas[start:]]
    if smooth:
        for i in range(1, len(loss)):
            loss[i] = (loss[i-1]*i+loss[i])/(i+1)
    plt.plot([i+start for i in range(len(loss))], loss)
    plt.xlim(0,len(loss)*11//10)
    save_path = "data/{0}.jpg".format(vis_to_lossname[loss_index])
    plt.savefig(save_path)
    print("loss figure saved in {0}".format(save_path))


def visual_data_from_file(file_path="datasets/wines.json"):
    with open(file_path, "r") as f:
        labels = []
        datas = json.load(f)

        annos = datas["Data"]["LSJZList"][::-1][-29:]
        for i in annos:
            if i["LJJZ"] and i["FSRQ"]:
                print(i["FSRQ"], i["LJJZ"])
                labels.append(float(i["LJJZ"]))

    plt.plot([i for i in range(len(labels))], labels)
    # plt.ylim(1.1, 1.6)
    plt.savefig("data/history.png")


if __name__ == "__main__":
    filename = "chkpt/20210113.log"
    visual_data_from_file()
    for i in range(len(vis_to_lossname)):
        visual_loss_from_file(filename, i, 100, False)

