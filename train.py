# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 下午2:49
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : train.py
# @Software: PyCharm

from model import Model
from datasets.wineDataset import WineDataset
import torch.utils.data.dataloader as Dataloader
import torch.optim as optim
import os
from torch.autograd import Variable
import torch
from logger import Logger
from cosine_lr_scheduler import CosineDecayLR
from loss import WineLoss

def setup_seed(seed=19960715):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed) #gpu
    torch.backends.cudnn.deterministic=True # cudnn

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    log = Logger()
    epochs = 90
    ### model ###
    net = Model().cuda()
    log.write(str(net))

    ### dataset ###
    train_dataset = WineDataset()
    train_dataloader = Dataloader.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    ### optim ###
    lr = 1e-4
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
    criterion = WineLoss().cuda()

    ### scheduler ###
    scheduler = CosineDecayLR(optimizer=optimizer, T_max=epochs*len(train_dataloader),lr_init=lr,
                              lr_min=lr*0.01,warmup=5*len(train_dataloader))

    net.train()
    for epoch in range(epochs):
        for i, (x, label) in enumerate(train_dataloader):
            scheduler.step(len(train_dataloader) * epoch + i)

            x, label = Variable(x.cuda()), Variable(label.cuda())
            pred = net(x)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log.write("epoch %3d, iter %2d, lr %2.10f, loss %2.4f"%
                      (epoch, i, optimizer.param_groups[0]['lr'], loss.item()))

            if loss <= .001:
                torch.save(net.state_dict(),"chkpt/weights.pth")
                log.write("save weights in chkpt/weights.pth")
                exit(0)
    torch.save(net.state_dict(),"chkpt/last_weights.pth")
    log.write("save weights in chkpt/weights.pth")

if __name__ == "__main__":
    setup_seed()
    train()