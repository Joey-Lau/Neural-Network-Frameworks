import keras
import numpy as np
from keras import layers, models
from keras.regularizers import l2
from keras.applications.mobilenet import relu6,DepthwiseConv2D
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling2D, Input, LeakyReLU, BatchNormalization
from keras.layers import Convolution2D as Conv2D
from keras import backend as K
from functools import wraps # 可以将原函数对象的指定属性复制给包装函数对象, 默认有 __module__、__name__、__doc__,或者通过参数选择
import time

@wraps(layers.Conv2D)
def my_conv(*args,**kwargs):
    mob_kwargs={'kernel_regularizer':l2(1e-6)}
    mob_kwargs['padding']='same'
    mob_kwargs['kernel_size']=(2,2) if kwargs.get('strides')==(2,2) else (1,1)
    mob_kwargs['kernel_initializer']= keras.initializers.glorot_uniform(seed=0)
    mob_kwargs.update(kwargs)
    return layers.Conv2D(*args,**mob_kwargs)
def bottleneck(input,filter,stride,t,con=False):
    # channel=K.int_shape(input)[-1]*t
    x=my_conv(filters=filter,strides=(1,1))(input)      #channel设置
    x=layers.BatchNormalization(axis=-1)(x)
    x=Activation(relu6)(x)

    x=DepthwiseConv2D((3,3),strides=(stride,stride),depthwise_regularizer=l2(1e-6),padding='same',depth_multiplier=t)(x)
    x=layers.BatchNormalization(axis=-1)(x)
    x=Activation(relu6)(x)

    x=my_conv(filters=filter,strides=(1,1))(x)
    x=layers.BatchNormalization(axis=-1)(x)
    if con:
        x=layers.add([x,input])
    return x
def residual_bottleneck(input,filter,stride,t,n):
    x=bottleneck(input,filter,stride,t)
    for i in range(1,n):
        x=bottleneck(x,filter,1,t,con=True)   #stride=1?
    return x
def mobiletnet(input_shape,classes,weight_button=False):
    inputs=layers.Input(input_shape)
    x=my_conv(filters=32,strides=(2,2))(inputs)
    x=residual_bottleneck(x,16,1,1,1)
    x=residual_bottleneck(x,24,2,6,2)
    x=residual_bottleneck(x,32,2,6,3)
    x=residual_bottleneck(x,64,2,6,4)
    x=residual_bottleneck(x,96,1,6,3)
    x=residual_bottleneck(x,160,2,6,3)
    x=residual_bottleneck(x,320,1,6,1)  # 147
    x=my_conv(filters=1280,strides=(1,1))(x)
    x=layers.AveragePooling2D((7,7),strides=(1,1))(x)
    x=my_conv(filters=classes,strides=(1,1))(x)
    x=Activation('softmax')(x)
    out=layers.Reshape((classes,))(x)
    model=keras.models.Model(inputs,out)
    if (weight_button):
        model.load_weights('mobiletnet_weights.h5')
    return model


