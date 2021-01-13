# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 下午3:14
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : loss.py
# @Software: PyCharm
import torch.nn as nn

class WineLoss(nn.Module):
    def __init__(self):
        super(WineLoss, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss()
    def forward(self, pred, label):
        loss = self.smoothl1(pred, label)
        return loss