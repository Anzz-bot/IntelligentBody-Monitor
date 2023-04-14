# encoding:utf-8
import os
import math
import random
import sys
import time
import threading
import numpy as np
import tensorflow as tf
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as mpcm

# sys.path.append('./SSD-Tensorflow/')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from window_dancheng import Ui_Win_traffic
# from ssdpersonDetect import *

class MainWindow(Ui_Win_traffic, QMainWindow):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setObjectName("mainWindow")
        self.setStyleSheet("#mainWindow{border-image:url(bg.jpg)}")
        self.output_size = 480
        self.model_size = 640
        self.device = 'cpu'
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头


        


        def load_model(self):
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)   # 自动选用设备
            isess = tf.InteractiveSession(config = config)   # 创建session
            slim = tf.contrib.slim # slim作为一种轻量级的tensorflow库，使得模型的构建，训练，测试都变得更加简单。
            l_VOC_CLASS = [
                'aeroplane',   'bicycle', 'bird',  'boat',      'bottle',
                'bus',         'car',     'cat',   'chair',     'cow',
                'diningTable', 'dog',     'horse', 'motorbike', 'person',
                'pottedPlant', 'sheep',   'sofa',  'train',     'TV']







        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.stopEvent2 = threading.Event()
        self.stopEvent2.clear()
        # self.model = self.model_load(weights="weights/yolov5s.pt",
        #                              device=self.device)  # todo 指明模型加载的位置的设备
        self.labels = ""
        self.imgz = 640
        self.setupUi(self)
        self.CAM_OPEN_FLAG = False

        # -----------页面切换按钮-----------
        self.pushButton.clicked.connect(self.open_cam)
        self.pushButton_2.clicked.connect(self.show_video)
        self.pushButton_3.clicked.connect(self.show_camera)
        # -----------页面切换按钮-----------

        # -----------功能按钮-----------
       

        self.show_picture_page.setCurrentIndex(0)

        # -----------功能按钮-----------

    def show_photo(self):
        self.show_picture_page.setCurrentIndex(0)

    def show_video(self):
        # self.stopEvent.set()
        self.show_picture_page.setCurrentIndex(1)

    def show_camera(self):
        self.stopEvent2.set()
        self.show_picture_page.setCurrentIndex(2)

    def open_cam(self):
        if not self.CAM_OPEN_FLAG:
            self.timer_camera = QTimer()  # 定义定时器
            #opencv read the camera
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3,640)
            self.cap.set(4,480)
            #create the QTimer to show the frame
            self.timer_camera.start(10)
            self.timer_camera.timeout.connect(self.show_frame)
            self.CAM_OPEN_FLAG = True
        else:
            # show messegebox
            reply = QMessageBox.warning(self, 'Error', 'The camera is opened!',
                                        QMessageBox.Yes, QMessageBox.Yes)
    # stop the camera
    def stop_cam(self):
        if self.CAM_OPEN_FLAG:
            # stop the QTmer
            self.timer_camera.stop()
            #clear the label
            self.label_page1.clear()
            self.label_page1.setText("Face Here")
            self.label_page1.setAlignment(Qt.AlignCenter)
            #release the camera
            self.cap.release()
            #turn off the flag
            self.CAM_OPEN_FLAG = False
        else:
            # show messegebox
            reply = QMessageBox.warning(self, 'Error', 'The camera is not opened!',
                                        QMessageBox.Yes, QMessageBox.Yes)

    def show_frame(self):
        if (self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            self.frame = cv2.flip(self.frame,1)
            if ret:
                self.frame = process_image(self.frame)
                cv2.imwrite("images/tmp/test_camera.jpg", self.frame)
                self.label.setPixmap(QPixmap("images/tmp/test_camera.jpg"))
    

    # def infer():

    
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
