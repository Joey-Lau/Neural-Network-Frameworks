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

def inception_module_a(x, f4):
    x_branch1 = x_branch2 = x_branch3 = x_branch4 = x
    x_branch1 = conv(x_branch1, filters=64)
    # print('x_branch1:', K.int_shape(x_branch1))

    x_branch2 = conv(x_branch2, filters=48)
    x_branch2 = conv(x_branch2, filters=64, kernel_size=(3, 3))
    # print('x_branch2:', K.int_shape(x_branch2))

    x_branch3 = conv(x_branch3, filters=64)
    x_branch3 = conv(x_branch3, filters=96, kernel_size=(3, 3))
    x_branch3 = conv(x_branch3, filters=96, kernel_size=(3, 3))
    # print('x_branch3:', K.int_shape(x_branch3))

    x_branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_branch4)
    x_branch4 = conv(x_branch4, filters=f4)
    # print('x_branch4:', K.int_shape(x_branch4))

    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3, x_branch4])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def inception_module_b(x):
    x_branch1 = x_branch2 = x_branch3 = x
    x_branch1 = conv(x_branch1,filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    # print('x_branch1:', K.int_shape(x_branch1))

    x_branch2 = conv(x_branch2, filters=64)
    x_branch2 = conv(x_branch2, filters=96, kernel_size=(3, 3))
    x_branch2 = conv(x_branch2, filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    # print('x_branch2:', K.int_shape(x_branch2))

    x_branch3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_branch3)
    # print('x_branch3:', K.int_shape(x_branch3))

    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def inception_module_c(x, f2, f3):
    x_branch1 = x_branch2 = x_branch3 = x_branch4 = x
    x_branch1 = conv(x_branch1, filters=192)
    # print('x_branch1:', K.int_shape(x_branch1))

    x_branch2 = conv(x_branch2, filters=f2)
    x_branch2 = conv(x_branch2, filters=f2, kernel_size=(1, 7))
    x_branch2 = conv(x_branch2, filters=192, kernel_size=(7, 1))
    # print('x_branch2:', K.int_shape(x_branch2))

    x_branch3 = conv(x_branch3, filters=f3)
    x_branch3 = conv(x_branch3, filters=f3, kernel_size=(1, 7))
    x_branch3 = conv(x_branch3, filters=f3, kernel_size=(7, 1))
    x_branch3 = conv(x_branch3, filters=f3, kernel_size=(1, 7))
    x_branch3 = conv(x_branch3, filters=192, kernel_size=(7, 1))
    # print('x_branch3:', K.int_shape(x_branch3))

    x_branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_branch4)
    x_branch4 = conv(x_branch4, filters=192)
    # print('x_branch4:', K.int_shape(x_branch4))

    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3, x_branch4])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def inception_module_d(x):
    x_branch1 = x_branch2 = x_branch3 = x
    x_branch1 = conv(x_branch1, filters=192)
    x_branch1 = conv(x_branch1, filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    # print('x_branch1:', K.int_shape(x_branch1))

    x_branch2 = conv(x_branch2, filters=192)
    x_branch2 = conv(x_branch2, filters=192, kernel_size=(1, 7))
    x_branch2 = conv(x_branch2, filters=192, kernel_size=(7, 1))
    x_branch2 = conv(x_branch2, filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    # print('x_branch2:', K.int_shape(x_branch2))

    x_branch3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_branch3)
    # print('x_branch3:', K.int_shape(x_branch3))

    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def inception_module_e(x):
    # x_branch1 = x_branch2 = x_branch3 = x_branch4 = x
    x_branch1 = conv(x, filters=320)
    # print('x_branch1:', K.int_shape(x_branch1))

    x_branch2 = conv(x, filters=384)
    # x_branch2_left = x_branch2_right = x_branch2
    x_branch2_left = conv(x_branch2, filters=384, kernel_size=(1, 3))
    x_branch2_right = conv(x_branch2, filters=384, kernel_size=(3, 1))
    x_branch2 = keras.layers.concatenate([x_branch2_left, x_branch2_right])
    # print('x_branch2:', K.int_shape(x_branch2))

    x_branch3 = conv(x, filters=448)
    x_branch3 = conv(x_branch3, filters=384, kernel_size=(3, 3))
    x_branch3_left = conv(x_branch3, filters=384, kernel_size=(1, 3))
    x_branch3_right = conv(x_branch3, filters=384, kernel_size=(3, 1))
    x_branch3 = keras.layers.concatenate([x_branch3_left, x_branch3_right])
    # print('x_branch3:', K.int_shape(x_branch3))

    x_branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x_branch4 = conv(x_branch4, filters=192)
    # print('x_branch4:', K.int_shape(x_branch4))

    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3, x_branch4])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def inception_v3(input_shape, classes): # input_shape=299 x 299 x 3
    x_input = Input(shape=input_shape)
    x = conv(x_input, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')   # (None, 149, 149, 32)
    x = conv(x, filters=32, kernel_size=(3, 3), padding='valid') # (None, 147, 147, 32)
    x = conv(x, filters=64, kernel_size=(3, 3)) # (None, 147, 147, 64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)  # (None, 73, 73, 64)
    x = BatchNormalization(axis=-1)(x)
    x = conv(x, filters=80, kernel_size=(3, 3), padding='valid') # (None, 71, 71, 80)
    x = conv(x, filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid') # (None, 35, 35, 192)
    x = inception_module_a(x, 32)  # (None, 35, 35, 256)
    x = inception_module_a(x, 64)  # (None, 35, 35, 288)
    x = inception_module_a(x, 64)  # (None, 35, 35, 288)
    x = inception_module_b(x)   # (None, 17, 17, 768)
    x = inception_module_c(x, f2=128, f3=128)   # (None, 17, 17, 768)
    x = inception_module_c(x, f2=160, f3=160)   # (None, 17, 17, 768)
    x = inception_module_c(x, f2=160, f3=160)   # (None, 17, 17, 768)
    x = inception_module_c(x, f2=192, f3=192)   # (None, 17, 17, 768)
    x = inception_module_d(x)   # (None, 8, 8, 1280)
    x = inception_module_e(x)   # (None, 8, 8, 2048)
    x = inception_module_e(x)   # (None, 8, 8, 2048)
    x = MaxPooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(x)  # (None, 1, 1, 2048)
    x = BatchNormalization(axis=-1)(x)
    x = conv(x, filters=1000)   # (None, 1, 1, 1000)
    x = Flatten()(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # end: (None, classes)
    # print('end:', K.int_shape(x))
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model


if __name__ == '__main__':
    inception_v3(input_shape=(299,299,3), classes=10)