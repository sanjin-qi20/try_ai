# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 1:08
# @Author  : Qisx
# @File    : work_2_train.py
# @Software: PyCharm

import os
import datetime

from work_2_CGAN import CGAN
from work_2_confug import MnistConfig
from work_2_Generator import MnistGenerator

def run_main():
    """
    这是主函数
    """
    cfg =  MnistConfig()
    cgan = CGAN(cfg)
    batch_size = 512
    #train_datagen = Cifar10Generator(int(batch_size/2))
    train_datagen = MnistGenerator(batch_size)
    cgan.train(train_datagen,100000,1,batch_size)


if __name__ == '__main__':
    run_main()