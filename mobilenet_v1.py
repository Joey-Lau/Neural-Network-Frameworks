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
    x=my_conv(**kwargs)(x)
    x=BatchNormalization(axis=-1)(x)
    # x=LeakyReLU(alpha=0.05)(x)
    x = Activation(relu6)(x)
    return x
def bottleneck(x, f, size, stride):
    x = DepthwiseConv2D(kernel_size=(size, size), strides=(stride, stride), padding='same', depth_multiplier=1,
                        depthwise_regularizer=keras.regularizers.l2(1e-6),
                        depthwise_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(relu6)(x)
    x = conv(x, filters=f, kernel_size=(3, 3), strides=(1, 1))
    return x
def mobilenet_v1(input_shape, classes): # (None, 224, 224, 32)
    x_input = keras.layers.Input(shape=input_shape)
    x = conv(x_input, filters=32, kernel_size=(3, 3), strides=(2, 2)) # (None, 112, 112, 32)
    x = bottleneck(x, f=32, size=3, stride=1) # (None, 112, 112, 32)
    x = conv(x, filters=64, kernel_size=(1, 1), strides=(1, 1)) # (None, 112, 112, 64)
    x = bottleneck(x, f=64, size=3, stride=2) # (None, 56, 56, 64)
    x = conv(x, filters=128, kernel_size=(1, 1), strides=(1, 1)) # (None, 56, 56, 128)
    x = bottleneck(x, f=128, size=3, stride=1) # (None, 56, 56, 128)
    x = conv(x, filters=128, kernel_size=(1, 1), strides=(1, 1))  # (None, 56, 56, 128)
    x = bottleneck(x, f=128, size=3, stride=2) # (None, 28, 28, 128)
    x = conv(x, filters=256, kernel_size=(1, 1), strides=(1, 1)) # (None, 28, 28, 256)
    x = bottleneck(x, f=256, size=3, stride=1) # (None, 28, 28, 256)
    x = conv(x, filters=256, kernel_size=(1, 1), strides=(1, 1)) # (None, 28, 28, 256)
    x = bottleneck(x, f=256, size=3, stride=2) # (None, 14, 14, 256)
    x = conv(x, filters=512, kernel_size=(1, 1), strides=(1, 1)) # (None, 14, 14, 512)
    for i in range(5):
        x = bottleneck(x, f=512, size=3, stride=1) # (None, 14, 14, 512)
    for i in range(5):
        x = conv(x, filters=512, kernel_size=(1, 1), strides=(1, 1)) # (None, 14, 14, 512)
    x = bottleneck(x, f=512, size=3, stride=2) # (None, 7, 7, 512)
    x = conv(x, filters=1024, kernel_size=(1, 1), strides=(1, 1)) # (None, 7, 7, 1024)
    x = bottleneck(x, f=1024, size=3, stride=1) # (None, 7, 7, 1024)
    x = conv(x, filters=1024, kernel_size=(1, 1), strides=(1, 1))  # (None, 7, 7, 1024)
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(x) # (None, 1, 1, 1024)
    x = BatchNormalization(axis=-1)(x)
    x = Flatten()(x)
    x = Dense(units=1024, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    # print(K.int_shape(x))
    model = keras.Model(inputs=x_input, outputs=x)
    return model


if __name__ == '__main__':
    mobilenet_v1(input_shape=(224,224,3), classes=20)