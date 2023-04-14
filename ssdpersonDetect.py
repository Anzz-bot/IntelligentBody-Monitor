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

#gpu_options = tf.GPUOptions(allow_growth=True)    # 允许使用GPU
#config = tf.ConfigProto(log_device_placement = False, gpu_options = gpu_options)
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)   # 自动选用设备
isess = tf.InteractiveSession(config = config)   # 创建session

slim = tf.contrib.slim # slim作为一种轻量级的tensorflow库，使得模型的构建，训练，测试都变得更加简单。


# l_VOC_CLASS = [
#                 'aeroplane',   'bicycle', 'bird',  'boat',      'bottle',
#                 'bus',         'car',     'cat',   'chair',     'cow',
#                 'diningTable', 'dog',     'horse', 'motorbike', 'person',
#                 'pottedPlant', 'sheep',   'sofa',  'train',     'TV'
# ]


l_VOC_CLASS = [
                'aeroplane',   'bicycle', 'bird',  'boat',      'bottle',
                'bus',         'car',     'cat',   'chair',     'cow',
                'diningTable', 'dog',     'horse', 'motorbike', 'person',
]

net_shape = (300,300)
data_format = 'NHWC' # [Number, height, width, color]

# 预处理,以tensorflow backend, 将输入图片大小改成 300*300, 作为下一步的输入
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input,
    None,
    None,
    net_shape,     # 300*300
    data_format,
    resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE
)
image_4d = tf.expand_dims(image_pre, 0)   # 将image_pre增加维度, 为什么这里要正增加维度

# 定义SSD模型结构
reuse = True if 'ssd_net' in locals() else None   # locals() 函数会以字典类型返回当前位置的全部局部变量。
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# 导入官方给出的SSD模型
ckpt_filename = './SSD-Tensorflow/checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)   # 把模型取出来交给会话isess

# 在网络模型结构中,提取搜索网格的位置
ssd_anchors = ssd_net.anchors(net_shape)   # 获取候选区域





###
# 该函数是根据有多少物种生成一个颜色列表, 不同的物种对应着不同的颜色
###
def colors_subselect(colors, num_classes=21):
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




colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)




###
# 获取检测结果, 对图片中检测到物体进行标记
# 参数 select_threshold : 可信度设置, 大于该值的都输出
# 参数 nms_threshold : nms的参数, 越小越能保证不出现重叠框
###
def process_image(img, select_threshold=0.2, nms_threshold=0.4, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})  

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    #print(rclasses)     # 对应的分类标签号
    #print(rscores)      # 对应的分类得分
    #print(rbboxes)      # 对应的框坐标

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)  # 进行nms处理 
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    #print(rclasses)    
    #print(rscores)     
    #print(rbboxes)
    bboxes_draw_on_img(img, rclasses, rscores, rbboxes, colors_plasma, thickness=2)     # 对img图片进行标记
    return img,rclasses,rscores


# img = cv2.imread("./test3.png")
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 颜色空间转换, jpg图像这里没有必要转
# #plt.imshow(process_image(img))

# cv2.imshow('test', process_image(img))
# cv2.waitKey()
# #print('end')

# cap = cv2.VideoCapture(0)
# while True:
    
#     ret,frame = cap.read()
#     if ret:
#         frame = process_image(frame)
#         cv2.imshow('test', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
        

































