# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 下午3:14
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : logger.py
# @Software: PyCharm
import sys
from time import strftime


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout

        self.log = open("chkpt//{0}.log".format(strftime("%Y%m%d")), "a")

    def write(self, message):
        self.terminal.write(message+"\n")
        self.log.write(message+"\n")




