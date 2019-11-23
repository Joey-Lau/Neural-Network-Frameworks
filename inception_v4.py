import numpy as np
import keras
from keras.layers import Add,Conv2D,Dense,Dropout,Flatten,Activation,AveragePooling2D,MaxPooling2D,Input,LeakyReLU,BatchNormalization,ZeroPadding2D
# from keras.models import Model,Sequential,Layer
from keras.regularizers import l2
from keras.initializers import glorot_uniform
# from keras.applications.mobilenet import relu6,DepthwiseConv2D
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
import time

from functools import wraps
@wraps(Conv2D)
def my_conv(*args,**kwargs):
    new_kwargs={'kernel_regularizer':l2(1e-6)}
    new_kwargs['padding']='same'
    new_kwargs['kernel_size']=(1,1)
    new_kwargs['strides']=(2,2) if kwargs.get('strides')==(2,2) else (1,1)
    new_kwargs['kernel_initializer']=keras.initializers.glorot_uniform(seed=0)
    new_kwargs.update(kwargs)
    return Conv2D(*args,**new_kwargs)
def conv(x,**kwargs):
    x=my_conv(**kwargs)(x)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)
    return x

def block(x, f):
    x_left = conv(x, filters=f, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    # print(K.int_shape(x_left))
    x_right = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    # print(K.int_shape(x_right))
    x = keras.layers.concatenate([x_left, x_right])
    return x
def stem(x):
    x = conv(x, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid') # (None, 149, 149, 32)
    x = conv(x, filters=32, kernel_size=(3, 3), padding='valid') # (None, 147, 147, 32)
    x = conv(x, filters=64, kernel_size=(3, 3)) # (None, 147, 147, 64)

    # x_left = conv(x, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid') # (None, 73, 73, 64)
    # x_right = conv(x, filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid') # (None, 73, 73, 96)
    # x = keras.layers.concatenate([x_left, x_right]) # (None, 73, 73, 160)
    x = block(x, f=96) # (None, 73, 73, 160)

    x_left = conv(x, filters=64)
    x_left = conv(x_left, filters=96, kernel_size=(3, 3), padding='valid') # (None, 71, 71, 96)
    x_right = conv(x, filters=64)
    x_right = conv(x_right, filters=64, kernel_size=(7, 1))
    x_right = conv(x_right, filters=64, kernel_size=(1, 7))
    x_right = conv(x_right, filters=96, kernel_size=(3, 3), padding='valid') # (None, 71, 71, 96)
    x = keras.layers.concatenate([x_left, x_right]) # (None, 71, 71, 192)
    x = block(x, f=192) # (None, 35, 35, 384)
    # print(K.int_shape(x))
    return x
def inception_a(x):  # (None, 35, 35, 384)
    x_branch1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    x_branch1 = conv(x_branch1, filters=96)
    x_branch2 = conv(x, filters=96)
    x_branch3 = conv(x, filters=64)
    x_branch3 = conv(x_branch3, filters=96, kernel_size=(3, 3))
    x_branch4 = conv(x, filters=64)
    x_branch4 = conv(x_branch4, filters=96, kernel_size=(3, 3))
    x_branch4 = conv(x_branch4, filters=96, kernel_size=(3, 3))
    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3, x_branch4])
    return x
def reduction_a(x):
    x_branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x_branch2 = conv(x, filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x_branch3 = conv(x, filters=256)
    x_branch3 = conv(x_branch3, filters=256, kernel_size=(3, 3))
    x_branch3 = conv(x_branch3, filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3])
    return x
def inception_b(x):
    x_branch1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    x_branch1 = conv(x_branch1, filters=128)
    x_branch2 = conv(x, filters=384)
    x_branch3 = conv(x, filters=192)
    x_branch3 = conv(x_branch3, filters=224, kernel_size=(1, 7))
    x_branch3 = conv(x_branch3, filters=256, kernel_size=(7, 1))
    x_branch4 = conv(x, filters=192)
    x_branch4 = conv(x_branch4, filters=192, kernel_size=(1, 7))
    x_branch4 = conv(x_branch4, filters=224, kernel_size=(7, 1))
    x_branch4 = conv(x_branch4, filters=224, kernel_size=(1, 7))
    x_branch4 = conv(x_branch4, filters=256, kernel_size=(7, 1))
    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3, x_branch4])
    return x
def reduction_b(x):
    x_branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x_branch2 = conv(x, filters=192)
    x_branch2 = conv(x_branch2, filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x_branch3 = conv(x, filters=256)
    x_branch3 = conv(x_branch3, filters=256, kernel_size=(1, 7))
    x_branch3 = conv(x_branch3, filters=320, kernel_size=(7, 1))
    x_branch3 = conv(x_branch3, filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3])
    return x
def inception_c(x):
    x_branch1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    x_branch1 = conv(x_branch1, filters=256)
    x_branch2 = conv(x, filters=256)
    x_branch3 = conv(x, filters=384)
    x_branch31 = conv(x_branch3, filters=256, kernel_size=(1, 3))
    x_branch32 = conv(x_branch3, filters=256, kernel_size=(3, 1))
    x_branch4 = conv(x, filters=384)
    x_branch4 = conv(x_branch4, filters=448, kernel_size=(1, 3))
    x_branch4 = conv(x_branch4, filters=512, kernel_size=(3, 1))
    x_branch41 = conv(x_branch4, filters=256, kernel_size=(1, 3))
    x_branch42 = conv(x_branch4, filters=256, kernel_size=(3, 1))
    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch31, x_branch32, x_branch41, x_branch42])
    return x
def inception_v4(input_shape, classes): # input_shape=299 x 299 x 3
    x_input = Input(shape=input_shape)
    x = stem(x_input) # (None, 35, 35, 384)
    for i in range(4):
        x = inception_a(x) # (None, 35, 35, 384)
    x = reduction_a(x) # (None, 17, 17, 1024)
    for i in range(7):
        x = inception_b(x) # (None, 17, 17, 1024)
    x = reduction_b(x)  # (None, 8, 8, 1536)
    for i in range(3):
        x = inception_c(x) # (None, 8, 8, 1536)
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(x)  # (None, 1, 1, 1536)
    x = BatchNormalization(axis=-1)(x)
    x = Flatten()(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # end: (None, classes)
    # print('end:', K.int_shape(x))
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model

if __name__ == '__main__':
    inception_v4(input_shape=(299,299,3), classes=10)