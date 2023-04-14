import argparse
from cProfile import label
import time
from pathlib import Path
import threading
import sys
import os
import cv2
# import torch
#print('detect.py', torch.__version__)
#print('detect.py', torch.__path__)

# import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import shutil
# from utils.torch_utils import select_device, load_classifier, time_synchronized

# from utils.datasets import *
# from utils.utils import *

import pyttsx3
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from window_xzc import Ui_Win_traffic
from ssdpersonDetect import *

class MainWindow(Ui_Win_traffic, QMainWindow):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setObjectName("mainWindow")
        self.setStyleSheet("#mainWindow{border-image:url(bg.jpg)}")
        self.output_size = 480
        self.model_size = 640
        self.img2predict = ""
        self.device = 'cpu'
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
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
        self.pushButton_2.clicked.connect(self.stop_cam)
        # self.pushButton_3.clicked.connect(self.show_camera)
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
    
    def warning_voice(self):
        self.timer_vol.start(1000)

    def voice(self):
        pyttsx3.speak('发现有人')


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

    def show_frame( self):
        if (self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            self.frame = cv2.flip(self.frame,1)
            if ret:
                self.frame,self.label,rscores= process_image(self.frame)
                cv2.imwrite("images/tmp/test_camera.jpg", self.frame)
                self.label_page1.setPixmap(QPixmap("images/tmp/test_camera.jpg"))
                self.data_log.append(str(rscores))
                if self.label != [] and (15 in self.label):
                    pyttsx3.speak('发现有人')


        



    # def infer():

    
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
