import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input,Lambda
from keras.layers import MaxPooling2D,Concatenate,UpSampling2D
from keras.layers import merge, concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D,BatchNormalization
from keras.models import Model


def conv2d_bn(x,filters,num_row,num_col,padding='same',stride=1,dilation_rate=1,relu=True):
    x = Conv2D(
        filters, (num_row, num_col),
        strides=(stride,stride),
        padding=padding,
        dilation_rate=(dilation_rate, dilation_rate),
        use_bias=False)(x)
    x = BatchNormalization(scale=False)(x)
    if relu:    
        x = Activation("relu")(x)
    return x


def BasicRFB(x,input_filters,output_filters,stride=1,map_reduce=8):
    input_filters_div = input_filters//map_reduce

    branch_0 = conv2d_bn(x,input_filters_div*2,1,1,stride=stride)
    branch_0 = conv2d_bn(branch_0,input_filters_div*2,3,3,relu=False)

    branch_1 = conv2d_bn(x,input_filters_div,1,1)
    branch_1 = conv2d_bn(branch_1,input_filters_div*2,3,3,stride=stride)
    branch_1 = conv2d_bn(branch_1,input_filters_div*2,3,3,dilation_rate=3,relu=False)
    
    branch_2 = conv2d_bn(x,input_filters_div,1,1)
    branch_2 = conv2d_bn(branch_2,(input_filters_div//2)*3,3,3)
    branch_2 = conv2d_bn(branch_2,input_filters_div*2,3,3,stride=stride)
    branch_2 = conv2d_bn(branch_2,input_filters_div*2,3,3,dilation_rate=5,relu=False)

    branch_3 = conv2d_bn(x,input_filters_div,1,1)
    branch_3 = conv2d_bn(branch_3,(input_filters_div//2)*3,1,7)
    branch_3 = conv2d_bn(branch_3,input_filters_div*2,7,1,stride=stride)
    branch_3 = conv2d_bn(branch_3,input_filters_div*2,3,3,dilation_rate=7,relu=False)

    out = concatenate([branch_0,branch_1,branch_2,branch_3],axis=-1)
    out = conv2d_bn(out,output_filters,1,1,relu=False)

    short = conv2d_bn(x,output_filters,1,1,stride=stride,relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out,short])
    out = Activation("relu")(out)
    return out

def BasicRFB_c(x,input_filters,output_filters,stride=1,map_reduce=8):
    input_filters_div = input_filters//map_reduce

    branch_0 = conv2d_bn(x,input_filters_div*2,1,1,stride=stride)
    branch_0 = conv2d_bn(branch_0,input_filters_div*2,3,3,relu=False)

    branch_1 = conv2d_bn(x,input_filters_div,1,1)
    branch_1 = conv2d_bn(branch_1,input_filters_div*2,3,3,stride=stride)
    branch_1 = conv2d_bn(branch_1,input_filters_div*2,3,3,dilation_rate=3,relu=False)
    
    branch_2 = conv2d_bn(x,input_filters_div,1,1)
    branch_2 = conv2d_bn(branch_2,(input_filters_div//2)*3,1,7)
    branch_2 = conv2d_bn(branch_2,input_filters_div*2,7,1,stride=stride)
    branch_2 = conv2d_bn(branch_2,input_filters_div*2,3,3,dilation_rate=7,relu=False)

    out = concatenate([branch_0,branch_1,branch_2],axis=-1)
    out = conv2d_bn(out,output_filters,1,1,relu=False)

    short = conv2d_bn(x,output_filters,1,1,stride=stride,relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out,short])
    out = Activation("relu")(out)
    return out

def BasicRFB_a(x,input_filters,output_filters,stride=1,map_reduce=8):
    input_filters_div = input_filters//map_reduce

    branch_0 = conv2d_bn(x,input_filters_div,1,1,stride=stride)
    branch_0 = conv2d_bn(branch_0,input_filters_div,3,3,relu=False)

    branch_1 = conv2d_bn(x,input_filters_div,1,1)
    branch_1 = conv2d_bn(branch_1,input_filters_div,3,1)
    branch_1 = conv2d_bn(branch_1,input_filters_div,3,3,dilation_rate=3,relu=False)
    
    branch_2 = conv2d_bn(x,input_filters_div,1,1)
    branch_2 = conv2d_bn(branch_2,input_filters_div,1,3)
    branch_2 = conv2d_bn(branch_2,input_filters_div,3,3,dilation_rate=3,relu=False)

    branch_3 = conv2d_bn(x,input_filters_div,1,1)
    branch_3 = conv2d_bn(branch_3,input_filters_div,3,1)
    branch_3 = conv2d_bn(branch_3,input_filters_div,3,3,dilation_rate=5,relu=False)
    
    branch_4 = conv2d_bn(x,input_filters_div,1,1)
    branch_4 = conv2d_bn(branch_4,input_filters_div,1,3)
    branch_4 = conv2d_bn(branch_4,input_filters_div,3,3,dilation_rate=5,relu=False)

    branch_5 = conv2d_bn(x,input_filters_div//2,1,1)
    branch_5 = conv2d_bn(branch_5,(input_filters_div//4)*3,1,3)
    branch_5 = conv2d_bn(branch_5,input_filters_div,3,1,stride=stride)
    branch_5 = conv2d_bn(branch_5,input_filters_div,3,3,dilation_rate=7,relu=False)

    branch_6 = conv2d_bn(x,input_filters_div//2,1,1)
    branch_6 = conv2d_bn(branch_6,(input_filters_div//4)*3,3,1)
    branch_6 = conv2d_bn(branch_6,input_filters_div,1,3,stride=stride)
    branch_6 = conv2d_bn(branch_6,input_filters_div,3,3,dilation_rate=7,relu=False)

    out = concatenate([branch_0,branch_1,branch_2,branch_3,branch_4,branch_5,branch_6],axis=-1)
    out = conv2d_bn(out,output_filters,1,1,relu=False)

    short = conv2d_bn(x,output_filters,1,1,stride=stride,relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out,short])
    out = Activation("relu")(out)
    return out


def Normalize(net):
    branch_0 = conv2d_bn(net["conv4_3"],256,1,1)
    branch_1 = conv2d_bn(net['fc7'],256,1,1)
    branch_1 = UpSampling2D()(branch_1)
    out = concatenate([branch_0,branch_1],axis=-1)
    out = BasicRFB_a(out,512,512)
    return out

def backbone(input_tensor):
    #----------------------------主干特征提取网络开始---------------------------#
    # RFB结构,net字典
    net = {} 
    # Block 1
    net['input'] = input_tensor
    # 300,300,3 -> 150,150,64
    net['conv1_1'] = Conv2D(64, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = Conv2D(64, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool1')(net['conv1_2'])

    
    # Block 2
    # 150,150,64 -> 75,75,128
    net['conv2_1'] = Conv2D(128, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = Conv2D(128, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool2')(net['conv2_2'])
    # Block 3
    # 75,75,128 -> 38,38,256
    net['conv3_1'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool3')(net['conv3_3'])
    # Block 4
    # 38,38,256 -> 19,19,512
    net['conv4_1'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool4')(net['conv4_3'])
    # Block 5
    # 19,19,512 -> 19,19,512
    net['conv5_1'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                name='pool5')(net['conv5_3'])
    # FC6
    # 19,19,512 -> 19,19,1024
    net['fc6'] = Conv2D(1024, kernel_size=(3,3), dilation_rate=(6, 6),
                                     activation='relu', padding='same',
                                     name='fc6')(net['pool5'])

    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    # 19,19,1024 -> 19,19,1024
    net['fc7'] = Conv2D(1024, kernel_size=(1,1), activation='relu',
                               padding='same', name='fc7')(net['fc6'])

    net['norm'] = Normalize(net)

    net['rfb_1'] = BasicRFB(net['fc7'],1024,1024)

    net['rfb_2'] = BasicRFB(net['rfb_1'],1024,512,stride=2)

    net['rfb_3'] = BasicRFB(net['rfb_2'],512,256,stride=2)

    net['conv6_1'] = conv2d_bn(net['rfb_3'],128,1,1)

    net['conv6_2'] = conv2d_bn(net['conv6_1'],256,3,3,padding="valid")

    net['conv7_1'] = conv2d_bn(net['conv6_2'],128,1,1)

    net['conv7_2'] = conv2d_bn(net['conv7_1'],256,3,3,padding="valid")
    return net