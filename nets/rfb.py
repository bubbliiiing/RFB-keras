from keras.layers import (Activation, Concatenate, Conv2D, Flatten, Input,
                          Reshape)
from keras.models import Model

from nets.backbone import backbone


def RFB300(input_shape, num_classes=21):
    #---------------------------------#
    #   典型的输入大小为[300,300,3]
    #---------------------------------#
    input_tensor = Input(shape=input_shape)
    
    # net变量里面包含了整个RFB的结构，通过层名可以找到对应的特征层
    net = backbone(input_tensor)
    
    #-----------------------将提取到的主干特征进行处理---------------------------#
    # 对conv4_3的通道进行l2标准化处理 
    # 38,38,512
    num_anchors = 6
    # 预测框的处理
    # num_anchors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['norm_mbox_loc']        = Conv2D(num_anchors * 4, kernel_size=(3,3), padding='same', name='norm_mbox_loc')(net['norm'])
    net['norm_mbox_loc_flat']   = Flatten(name='norm_mbox_loc_flat')(net['norm_mbox_loc'])
    # num_anchors表示每个网格点先验框的数量，num_classes是所分的类
    net['norm_mbox_conf']       = Conv2D(num_anchors * num_classes, kernel_size=(3,3), padding='same',name='norm_mbox_conf')(net['norm'])
    net['norm_mbox_conf_flat']  = Flatten(name='norm_mbox_conf_flat')(net['norm_mbox_conf'])

    # 对rfb_1层进行处理 
    # 19,19,1024
    num_anchors = 6
    # 预测框的处理
    # num_anchors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['rfb_1_mbox_loc']       = Conv2D(num_anchors * 4, kernel_size=(3,3),padding='same',name='rfb_1_mbox_loc')(net['rfb_1'])
    net['rfb_1_mbox_loc_flat']  = Flatten(name='rfb_1_mbox_loc_flat')(net['rfb_1_mbox_loc'])
    # num_anchors表示每个网格点先验框的数量，num_classes是所分的类
    net['rfb_1_mbox_conf']      = Conv2D(num_anchors * num_classes, kernel_size=(3,3),padding='same',name='rfb_1_mbox_conf')(net['rfb_1'])
    net['rfb_1_mbox_conf_flat'] = Flatten(name='rfb_1_mbox_conf_flat')(net['rfb_1_mbox_conf'])
    
    # 对rfb_2进行处理
    # 10,10,512
    num_anchors = 6
    # 预测框的处理
    # num_anchors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['rfb_2_mbox_loc']       = Conv2D(num_anchors * 4, kernel_size=(3,3), padding='same',name='rfb_2_mbox_loc')(net['rfb_2'])
    net['rfb_2_mbox_loc_flat']  = Flatten(name='rfb_2_mbox_loc_flat')(net['rfb_2_mbox_loc'])
    # num_anchors表示每个网格点先验框的数量，num_classes是所分的类
    net['rfb_2_mbox_conf']      = Conv2D(num_anchors * num_classes, kernel_size=(3,3), padding='same',name='rfb_2_mbox_conf')(net['rfb_2'])
    net['rfb_2_mbox_conf_flat'] = Flatten(name='rfb_2_mbox_conf_flat')(net['rfb_2_mbox_conf'])

    # 对rfb_3进行处理
    # 5,5,256
    num_anchors = 6
    # 预测框的处理
    # num_anchors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['rfb_3_mbox_loc']       = Conv2D(num_anchors * 4, kernel_size=(3,3), padding='same',name='rfb_3_mbox_loc')(net['rfb_3'])
    net['rfb_3_mbox_loc_flat']  = Flatten(name='rfb_3_mbox_loc_flat')(net['rfb_3_mbox_loc'])
    # num_anchors表示每个网格点先验框的数量，num_classes是所分的类
    net['rfb_3_mbox_conf']      = Conv2D(num_anchors * num_classes, kernel_size=(3,3), padding='same',name='rfb_3_mbox_conf')(net['rfb_3'])
    net['rfb_3_mbox_conf_flat'] = Flatten(name='rfb_3_mbox_conf_flat')(net['rfb_3_mbox_conf'])

    # 对conv6_2进行处理
    # 3,3,256
    num_anchors = 4
    # 预测框的处理
    # num_anchors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv6_2_mbox_loc']         = Conv2D(num_anchors * 4, kernel_size=(3,3), padding='same',name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc_flat']    = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # num_anchors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv6_2_mbox_conf']        = Conv2D(num_anchors * num_classes, kernel_size=(3,3), padding='same',name='conv6_2_mbox_conf')(net['conv6_2'])
    net['conv6_2_mbox_conf_flat']   = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])

    # 对conv7_2进行处理
    # 1,1,256
    num_anchors = 4
    # 预测框的处理
    # num_anchors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv7_2_mbox_loc']         = Conv2D(num_anchors * 4, kernel_size=(3,3), padding='same',name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc_flat']    = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    # num_anchors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv7_2_mbox_conf']        = Conv2D(num_anchors * num_classes, kernel_size=(3,3), padding='same',name='conv7_2_mbox_conf')(net['conv7_2'])
    net['conv7_2_mbox_conf_flat']   = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])
    
    # 将所有结果进行堆叠
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['norm_mbox_loc_flat'],
                                                            net['rfb_1_mbox_loc_flat'],
                                                            net['rfb_2_mbox_loc_flat'],
                                                            net['rfb_3_mbox_loc_flat'],
                                                            net['conv6_2_mbox_loc_flat'],
                                                            net['conv7_2_mbox_loc_flat']])
                                                                    
    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['norm_mbox_conf_flat'],
                                                            net['rfb_1_mbox_conf_flat'],
                                                            net['rfb_2_mbox_conf_flat'],
                                                            net['rfb_3_mbox_conf_flat'],
                                                            net['conv6_2_mbox_conf_flat'],
                                                            net['conv7_2_mbox_conf_flat']])
    # 11620,4
    net['mbox_loc']     = Reshape((-1, 4), name='mbox_loc_final')(net['mbox_loc'])
    # 11620,21
    net['mbox_conf']    = Reshape((-1, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf']    = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    # 11620,25
    net['predictions']  = Concatenate(axis =-1, name='predictions')([net['mbox_loc'], net['mbox_conf']])

    model = Model(net['input'], net['predictions'])
    return model
