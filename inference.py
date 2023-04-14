# encoding:utf-8
import os
import math
import random
import sys
import time

import numpy as np
import tensorflow as tf
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as mpcm

sys.path.append('./SSD-Tensorflow/')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

l_VOC_CLASS = [
                'aeroplane',   'bicycle', 'bird',  'boat',      'bottle',
                'bus',         'car',     'cat',   'chair',     'cow',
                'diningTable', 'dog',     'horse', 'motorbike', 'person',
]

class SSDInference():
    def __init__(self) -> None:
        self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)   # 自动选用设备
        self.isess = tf.InteractiveSession(config = self.config)   # 创建session
        self.slim = tf.contrib.slim # slim作为一种轻量级的tensorflow库，使得模型的构建，训练，测试都变得更加简单。


        self.net_shape = (300,300)
        self.data_format = 'NHWC' # [Number, height, width, color]


        # 预处理,以tensorflow backend, 将输入图片大小改成 300*300, 作为下一步的输入
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            img_input,
            None,
            None,
            net_shape,     # 300*300
            data_format,
            resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE
        )
        image_4d = tf.expand_dims(image_pre, 0)   # 将image_pre增加维度, 为什么这里要正增加维度






    def load_model(self):
        pass
    



    ###
    # 该函数是根据有多少物种生成一个颜色列表, 不同的物种对应着不同的颜色
    ###
    def colors_subselect(self,colors, num_classes=21):
        dt = len(colors) // num_classes
        sub_colors = []
        for i in range(num_classes):
            color = colors[i*dt]
            if isinstance(color[0], float):
                sub_colors.append([int(c * 255) for c in color])
            else:
                sub_colors.append([c for c in color])
        return sub_colors


    ###
    # 该函数根据网络输出结果对进行进行标记
    # 参数说明:
    #          img       要标记的图片  
    #          classes   检测到的物体的类别
    #          scores    检测到的物体的置信度
    #          bboxes    检测到的物体的坐标信息
    #          colors    物种颜色列表, 不同的物种对应不同的颜色
    ###
    def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
        shape = img.shape
        for i in range(bboxes.shape[0]):
            if (classes[i] == 15):  # 是person才标记,其他的不标记
                bbox = bboxes[i]
                color = colors[classes[i]]
                # Draw bounding box...
                p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))   # 左上角(y_min, x_min) , 为什么要这样缩放
                p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))   # 右下角(y_max, x_max) , 为什么要这样缩放
                cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
                # Draw text...
                s = '%s/%.3f' % (l_VOC_CLASS[int(classes[i]) - 1], scores[i])    
                p1 = (p1[0] - 5, p1[1])
                cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)




    def infer(self):
        pass