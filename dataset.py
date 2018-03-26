#!/usr/bin/python3

import tensorflow as tf
import matplotlib.image as mpimg # mpimg 用于读取图片
import cv2 # 机器视觉库，安装请用pip3 install opencv-python
import numpy as np # 数值计算库
import os # 系统库
from random import shuffle # 随机数据库 
from tqdm import tqdm # 输出进度库
import matplotlib.pyplot as plt # 常用画图库
import tensorflow as tf
from scipy import misc


class DataSet(object):

    def __init__(self,size):
        self.img_size = size

    def __getlabel(self,img):
        word_label = img.split('.')[0]
        if word_label == 'dog':
            label = 0
        else:
            label = 1
        return label
    
    
    def create_train_set(self,dir):
        training_data = []
        training_label = []
        for img in tqdm(os.listdir(dir)):
            y_ = self.__getlabel(img)
            path = os.path.join(dir, img)
            img = cv2.imread(path)  # 读入RGB
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            x_ = cv2.resize(img,(self.img_size,self.img_size),interpolation=cv2.INTER_CUBIC)  # 将图片变成统一大小
            #print(x_.shape)
            training_data.append(x_)
            training_label.append(y_)
        return training_data,training_label

