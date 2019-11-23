import keras
import numpy as np
from keras import layers, models
from keras.regularizers import l2
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.layers import Add,Conv2D,Dense,Dropout,Flatten,AveragePooling2D,MaxPooling2D,Input,LeakyReLU,BatchNormalization,ZeroPadding2D,Activation
from keras import backend as K
import time
from functools import wraps
@wraps(Conv2D)
def my_conv(*args,**kwargs):
    new_kwargs={'kernel_regularizer':l2(1e-6)}
    new_kwargs['padding']='same'
    new_kwargs['kernel_size']=(1, 1)
    new_kwargs['strides']=(2, 2) if kwargs.get('strides')==(2, 2) else (1, 1)
    new_kwargs['kernel_initializer']=keras.initializers.glorot_uniform(seed=0)
    new_kwargs.update(kwargs)
    return Conv2D(*args,**new_kwargs)
def conv(x,**kwargs):
    x = my_conv(**kwargs)(x)
    x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    x = Activation(relu6)(x)
    return x

def bottleneck(x, f, stride, dm, button=False): # stride=1,True; stride=2,False
    x_short = x
    x = conv(x, filters=f, strides=(1, 1))
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(stride, stride), padding='same', depth_multiplier=dm,
                        depthwise_regularizer=keras.regularizers.l2(1e-6),
                        depthwise_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(relu6)(x)
    x = my_conv(filters=f, strides=(1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    if button:
        x = keras.layers.Add()([x, x_short])
    x = LeakyReLU(alpha=0.05)(x)
    return x
def my_bottleneck(x, f, stride, dm, n):
    x = bottleneck(x, f, stride, dm)
    for i in range(n):
        x = bottleneck(x, f, 1, dm, button=True) # stride=1
    return x
def mobilenet_v2(input_shape, classes):
    x_input = Input(shape=input_shape)
    x = conv(x_input, filters=32, strides=(2, 2)) # (None, 112, 112, 32)
    x = my_bottleneck(x, 16, 1, 1, 1) # (None, 112, 112, 16)
    x = my_bottleneck(x, 24, 2, 6, 2) # (None, 56, 56, 24)
    x = my_bottleneck(x, 32, 2, 6, 3) # (None, 28, 28, 32)
    x = my_bottleneck(x, 64, 1, 6, 4) # (None, 28, 28, 64)
    x = my_bottleneck(x, 96, 2, 6, 3) # (None, 14, 14, 96)
    x = my_bottleneck(x, 160, 2, 6, 3) # (None, 7, 7, 160)
    x = my_bottleneck(x, 320, 1, 6, 3) # (None, 7, 7, 320)
    x = conv(x, filters=1280) # (None, 7, 7, 1280)
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(x) # (None, 1, 1, 1280)
    x = conv(x, filters=classes) # (None, 1, 1, 20)
    print(K.int_shape(x))
    x = Activation('softmax')(x)
    out = layers.Reshape((classes,))(x) # (None, 20)
    print(K.int_shape(out))
    model = keras.models.Model(x_input, out)
    return model


if __name__ == '__main__':
    mobilenet_v2((224,224,3),20)
