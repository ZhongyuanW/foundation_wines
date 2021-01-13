# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 下午3:42
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : demo.py
# @Software: PyCharm

import torch
from model import Model
import os
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

def get_riqi(today, count=30):
    i = 0
    today = datetime.strptime(today,"%Y-%m-%d")
    riqi_list=[]
    riqi_list.append(str(date.strftime(today,"%Y-%m-%d")))
    while(i<=count):
        i+=1
        riqi = today + timedelta(days=i)
        riqi = date.strftime(riqi,"%Y-%m-%d")
        riqi_list.append(str(riqi))
    return riqi_list

def demo(dates):
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    net = Model().cuda()
    weight = torch.load("chkpt/last_weights.pth")
    net.load_state_dict(weight)
    net.eval()
    base_date = datetime.strptime("2015-09-16","%Y-%m-%d")
    preds = []
    with torch.no_grad():
        for date in dates:
            print(date, end=" ")
            x = (datetime.strptime(date,"%Y-%m-%d") - base_date).days
            x = torch.FloatTensor([[[[x]]]]).cuda()
            pred = net(x)
            pred = pred.cpu().squeeze(-1).squeeze(-1).item()
            preds.append(pred)
            print(pred)

    plt.plot([i+1 for i in range(len(preds))], preds)
    plt.savefig("data/pred.png")




if __name__ == "__main__":
    dates = get_riqi("2020-12-11",30)
    demo(dates)
