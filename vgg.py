import numpy as np
import keras
from keras.layers import Add,Conv2D,Dense,Dropout,Flatten,Activation,MaxPooling2D,Input,LeakyReLU,BatchNormalization,ZeroPadding2D
from keras.models import Model,Sequential,Layer
from keras.regularizers import l2
from keras.initializers import glorot_uniform
from keras.applications.mobilenet import relu6,DepthwiseConv2D
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
import time

from functools import wraps
@wraps(Conv2D)
def my_conv(*args,**kwargs):
    new_kwargs={'kernel_regularizer':l2(1e-6)}
    new_kwargs['padding']='same'
    new_kwargs['strides']=(2,2) if kwargs.get('strides')==(2,2) else (1,1)
    new_kwargs['kernel_initializer']=keras.initializers.glorot_uniform(seed=0)
    new_kwargs.update(kwargs)
    return Conv2D(*args,**new_kwargs)
def conv(x,**kwargs):
    x=my_conv(**kwargs)(x)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)
    return x

def vgg16(input_shape, classes):     # 224 x 224 x 3    13 Conv layer and 3 Dense layer
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(x_input)  # padding=3     230 x 230 x 3
    # print(K.int_shape(x))
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',     # 228 x 228 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),  # 施加在权重上的正则项
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # 权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器
    # print(K.int_shape(x))
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',      # 228 x 228 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)   # 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 114 x 114 x 64

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 114 x 114 x 128
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 114 x 114 x 128
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 57 x 57 x 128

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 57 x 57 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 57 x 57 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 57 x 57 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 28 x 28 x 256
    # print(K.int_shape(x))

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 28 x 28 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 28 x 28 x 512
           kernel_regularizer=keras.regularizers.l2(1e-6),
           kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 28 x 28 x 512
           kernel_regularizer=keras.regularizers.l2(1e-6),
           kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 14 x 14 x 512

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 14 x 14 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 14 x 14 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 14 x 14 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 7 x 7 x 512

    x = keras.layers.Flatten()(x)                                                   # 4096
    x = Dense(units=4096, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)    # 避免ReLU可能出现的神经元“死亡”现象
    x = Dense(units=4096, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(units=1000, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)

    model = keras.Model(inputs=x_input, outputs=x)
    return  model

def vgg19(input_shape, classes):    # 224 x 224 x 3    16 Conv layer and 3 Dense layer
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(x_input)  # padding=3     230 x 230 x 3
    # print(K.int_shape(x))
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',     # 228 x 228 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',      # 228 x 228 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 114 x 114 x 64

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 114 x 114 x 128
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 114 x 114 x 128
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 57 x 57 x 128

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 57 x 57 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 57 x 57 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 57 x 57 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 57 x 57 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 28 x 28 x 256
    # print(K.int_shape(x))

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 28 x 28 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 28 x 28 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 28 x 28 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 28 x 28 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 14 x 14 x 512

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 14 x 14 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 14 x 14 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 14 x 14 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 14 x 14 x 512
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 7 x 7 x 512

    x = keras.layers.Flatten()(x)  # 4096
    x = Dense(units=4096, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)  # 避免ReLU可能出现的神经元“死亡”现象
    x = Dense(units=4096, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(units=1000, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)

    model = keras.Model(inputs=x_input, outputs=x)
    return model

if __name__ == '__main__':
    vgg16(input_shape=(224, 224, 3), classes=10)
    vgg19(input_shape=(224, 224, 3), classes=10)