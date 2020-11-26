from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.preprocessing import image
from nets.rfb import rfb300
from nets.rfb_training import MultiboxLoss,Generator
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import cv2
import keras
import os
import sys

 
#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    log_dir = "logs/"
    annotation_path = '2007_train.txt'
    
    NUM_CLASSES = 21
    input_shape = (300, 300, 3)
    priors = get_anchors()
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    model = rfb300(input_shape, num_classes=NUM_CLASSES)
    
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    model.load_weights("model_data/rfb_weights.h5", by_name=True, skip_mismatch=True)
    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    BATCH_SIZE = 4
    gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]),NUM_CLASSES)
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    for i in range(21):
        model.layers[i].trainable = False
    if True:
        model.compile(optimizer=Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
                        loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE/2,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=15, 
                initial_epoch=0,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    if True:
        model.compile(optimizer=Adam(lr=2e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
                        loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE/2,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=30, 
                initial_epoch=15,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    for i in range(21):
        model.layers[i].trainable = True
    if True:
        model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=50, 
                initial_epoch=30,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
