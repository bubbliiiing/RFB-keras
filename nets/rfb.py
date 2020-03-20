import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import merge, concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model
from nets.backbone import backbone
from nets.rfb_layers import PriorBox


def rfb300(input_shape, num_classes=21):
    # 300,300,3
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    # RFB结构,net字典
    net = backbone(input_tensor)
    #-----------------------将提取到的主干特征进行处理---------------------------#
    # 对norm后的结果进行处理 38,38,512
    net['norm'] = net['norm']
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['norm_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same', name='norm_mbox_loc')(net['norm'])
    net['norm_mbox_loc_flat'] = Flatten(name='norm_mbox_loc_flat')(net['norm_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['norm_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='norm_mbox_conf')(net['norm'])
    net['norm_mbox_conf_flat'] = Flatten(name='norm_mbox_conf_flat')(net['norm_mbox_conf'])
    priorbox = PriorBox(img_size, 21.0,max_size = 45.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='norm_mbox_priorbox')
    net['norm_mbox_priorbox'] = priorbox(net['norm'])
    
    # 对rfb_1层进行处理 
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['rfb_1_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3),padding='same',name='rfb_1_mbox_loc')(net['rfb_1'])
    net['rfb_1_mbox_loc_flat'] = Flatten(name='rfb_1_mbox_loc_flat')(net['rfb_1_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['rfb_1_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3),padding='same',name='rfb_1_mbox_conf')(net['rfb_1'])
    net['rfb_1_mbox_conf_flat'] = Flatten(name='rfb_1_mbox_conf_flat')(net['rfb_1_mbox_conf'])

    priorbox = PriorBox(img_size, 45.0, max_size=99.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='rfb_1_mbox_priorbox')
    net['rfb_1_mbox_priorbox'] = priorbox(net['rfb_1'])

    # 对rfb_2进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='rfb_2_mbox_loc')(net['rfb_2'])
    net['rfb_2_mbox_loc'] = x
    net['rfb_2_mbox_loc_flat'] = Flatten(name='rfb_2_mbox_loc_flat')(net['rfb_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='rfb_2_mbox_conf')(net['rfb_2'])
    net['rfb_2_mbox_conf'] = x
    net['rfb_2_mbox_conf_flat'] = Flatten(name='rfb_2_mbox_conf_flat')(net['rfb_2_mbox_conf'])

    priorbox = PriorBox(img_size, 99.0, max_size=153.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='rfb_2_mbox_priorbox')
    net['rfb_2_mbox_priorbox'] = priorbox(net['rfb_2'])

    # 对rfb_3进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='rfb_3_mbox_loc')(net['rfb_3'])
    net['rfb_3_mbox_loc'] = x
    net['rfb_3_mbox_loc_flat'] = Flatten(name='rfb_3_mbox_loc_flat')(net['rfb_3_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='rfb_3_mbox_conf')(net['rfb_3'])
    net['rfb_3_mbox_conf'] = x
    net['rfb_3_mbox_conf_flat'] = Flatten(name='rfb_3_mbox_conf_flat')(net['rfb_3_mbox_conf'])

    priorbox = PriorBox(img_size, 153.0, max_size=207.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='rfb_3_mbox_priorbox')
    net['rfb_3_mbox_priorbox'] = priorbox(net['rfb_3'])

    # 对conv6_2进行处理
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc'] = x
    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv6_2_mbox_conf')(net['conv6_2'])
    net['conv6_2_mbox_conf'] = x
    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])

    priorbox = PriorBox(img_size, 207.0, max_size=261.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])

    # 对conv7_2进行处理
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc'] = x
    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv7_2_mbox_conf')(net['conv7_2'])
    net['conv7_2_mbox_conf'] = x
    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])
    
    priorbox = PriorBox(img_size, 261.0, max_size=315.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')

    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])

    # 将所有结果进行堆叠
    net['mbox_loc'] = concatenate([net['norm_mbox_loc_flat'],
                             net['rfb_1_mbox_loc_flat'],
                             net['rfb_2_mbox_loc_flat'],
                             net['rfb_3_mbox_loc_flat'],
                             net['conv6_2_mbox_loc_flat'],
                             net['conv7_2_mbox_loc_flat']],
                            axis=1, name='mbox_loc')
    net['mbox_conf'] = concatenate([net['norm_mbox_conf_flat'],
                              net['rfb_1_mbox_conf_flat'],
                              net['rfb_2_mbox_conf_flat'],
                              net['rfb_3_mbox_conf_flat'],
                              net['conv6_2_mbox_conf_flat'],
                              net['conv7_2_mbox_conf_flat']],
                             axis=1, name='mbox_conf')
    net['mbox_priorbox'] = concatenate([net['norm_mbox_priorbox'],
                                  net['rfb_1_mbox_priorbox'],
                                  net['rfb_2_mbox_priorbox'],
                                  net['rfb_3_mbox_priorbox'],
                                  net['conv6_2_mbox_priorbox'],
                                  net['conv7_2_mbox_priorbox']],
                                  axis=1, name='mbox_priorbox')

    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4

    net['mbox_loc'] = Reshape((num_boxes, 4),name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes),name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax',name='mbox_conf_final')(net['mbox_conf'])

    net['predictions'] = concatenate([net['mbox_loc'],
                               net['mbox_conf'],
                               net['mbox_priorbox']],
                               axis=2, name='predictions')
                               
    model = Model(net['input'], net['predictions'])
    return model