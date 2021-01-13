# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 下午2:22
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : model.py
# @Software: PyCharm

import torch.nn as nn
import torch


class LinearLayer(nn.Module):
    def __init__(self, inplane, outplane):
        super(LinearLayer, self).__init__()
        self.linear = nn.Conv2d(inplane,outplane,1,bias=False)
        self.norm = nn.BatchNorm2d(outplane)
        self.acivation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.acivation(x)
        return x


class Model(nn.Module):
    def __init__(self, inplane=1, layers=(64, 128, 256, 512, 512, 512, 64)):
        super(Model, self).__init__()
        my_layers = []
        for i in layers:
            my_layers.append(LinearLayer(inplane, i))
            inplane = i
        self.layers = nn.Sequential(*my_layers)
        self.out = nn.Conv2d(inplane, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.layers(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    x = torch.randn((2, 1, 1, 1))
    net = Model()
    x = net(x)
    print(x)





