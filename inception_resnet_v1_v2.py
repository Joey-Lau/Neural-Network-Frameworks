import numpy as np
import keras
from keras.layers import Add,Conv2D,Dense,Dropout,Flatten,Activation,AveragePooling2D,MaxPooling2D,Input,LeakyReLU,BatchNormalization,ZeroPadding2D
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
def stem_v1(x):
    x = conv(x, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid') # (None, 149, 149, 32)
    x = conv(x, filters=32, kernel_size=(3, 3), padding='valid') # (None, 147, 147, 32)
    x = conv(x, filters=64, kernel_size=(3, 3)) # (None, 147, 147, 64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x) # (None, 73, 73, 64)
    x = conv(x, filters=80) # (None, 73, 73, 80)
    x = conv(x, filters=192, kernel_size=(3, 3), padding='valid') # (None, 71, 71, 192)
    x = conv(x, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid') # (None, 35, 35, 256)
    # print(K.int_shape(x))
    return x
def stem_v2(x):
    x = conv(x, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')  # (None, 149, 149, 32)
    x = conv(x, filters=32, kernel_size=(3, 3), padding='valid')  # (None, 147, 147, 32)
    x = conv(x, filters=64, kernel_size=(3, 3))  # (None, 147, 147, 64)
    x = block(x, f=96)  # (None, 73, 73, 160)
    x_left = conv(x, filters=64)
    x_left = conv(x_left, filters=96, kernel_size=(3, 3), padding='valid')  # (None, 71, 71, 96)
    x_right = conv(x, filters=64)
    x_right = conv(x_right, filters=64, kernel_size=(7, 1))
    x_right = conv(x_right, filters=64, kernel_size=(1, 7))
    x_right = conv(x_right, filters=96, kernel_size=(3, 3), padding='valid')  # (None, 71, 71, 96)
    x = keras.layers.concatenate([x_left, x_right])  # (None, 71, 71, 192)
    x = block(x, f=192)  # (None, 35, 35, 384)
    # print(K.int_shape(x))
    return x
def inception_resnet_a(x, f1, f2, f3):
    x_branch1 = x
    x_branch2_left = conv(x, filters=32)
    x_branch2_middle = conv(x, filters=32)
    x_branch2_middle = conv(x_branch2_middle, filters=32, kernel_size=(3, 3))
    x_branch2_right = conv(x, filters=32)
    x_branch2_right = conv(x_branch2_right, filters=f1, kernel_size=(3, 3))
    x_branch2_right = conv(x_branch2_right, filters=f2, kernel_size=(3, 3))
    x_branch2 = keras.layers.concatenate([x_branch2_left, x_branch2_middle, x_branch2_right])
    x_branch2 = conv(x_branch2, filters=f3)
    x = keras.layers.Add()([x_branch1, x_branch2])
    x = LeakyReLU(alpha=0.05)(x)
    return x
def reduction_a(x, f):
    x_branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x_branch2 = conv(x, filters=f, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x_branch3 = conv(x, filters=256)
    x_branch3 = conv(x_branch3, filters=256, kernel_size=(3, 3))
    x_branch3 = conv(x_branch3, filters=f, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3])
    x = LeakyReLU(alpha=0.05)(x)
    return x
def inception_resnet_b(x, f11, f21, f22, f2):
    x_branch1 = x
    x_branch2_left = conv(x, filters=f11)
    x_branch2_right = conv(x, filters=128)
    x_branch2_right = conv(x_branch2_right, filters=f21, kernel_size=(1, 7))
    x_branch2_right = conv(x_branch2_right, filters=f22, kernel_size=(7, 1))
    x_branch2 = keras.layers.Add()([x_branch2_left, x_branch2_right])
    x_branch2 = conv(x_branch2, filters=f2)
    x = keras.layers.Add()([x_branch1, x_branch2])
    x = LeakyReLU(alpha=0.05)(x)
    return x
def reduction_b(x, f1, f2, f3):
    x_branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x_branch2 = conv(x, filters=256)
    x_branch2 = conv(x_branch2, filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x_branch3 = conv(x, filters=256)
    x_branch3 = conv(x_branch3, filters=f1, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x_branch4 = conv(x, filters=256)
    x_branch4 = conv(x_branch4, filters=f2, kernel_size=(3, 3))
    x_branch4 = conv(x_branch4, filters=f3, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3, x_branch4])
    x = LeakyReLU(alpha=0.05)(x)
    return x
def inception_resnet_c(x, f1, f2, f3):
    x_branch1 = x
    x_branch2_left = conv(x, filters=192)
    x_branch2_right = conv(x, filters=192)
    x_branch2_right = conv(x_branch2_right, filters=f1, kernel_size=(1, 3))
    x_branch2_right = conv(x_branch2_right, filters=f2, kernel_size=(3, 1))
    x_branch2 = keras.layers.concatenate([x_branch2_left, x_branch2_right])
    x_branch2 = conv(x_branch2, filters=f3)
    x = keras.layers.Add()([x_branch1, x_branch2])
    x = LeakyReLU(alpha=0.05)(x)
    return x
def inception_resnet_v1(input_shape, classes): # input_shape=299 x 299 x 3
    x_input = Input(shape=input_shape)
    x = stem_v1(x_input) # (None, 35, 35, 256)
    for i in range(5):
        x = inception_resnet_a(x, f1=32, f2=32, f3=256) # (None, 35, 35, 256)
    x = reduction_a(x, f=320) # (None, 17, 17, 896)
    for i in range(10):
        x = inception_resnet_b(x, f11=128, f21=128, f22=128, f2=896) # (None, 17, 17, 896)
    x = reduction_b(x, f1=256, f2=256, f3=256)  # (None, 8, 8, 1792)
    for i in range(5):
        x = inception_resnet_c(x, f1=192, f2=192, f3=1792) # (None, 8, 8, 1792)
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(x)  # (None, 1, 1, 1792)
    x = BatchNormalization(axis=-1)(x)
    x = Flatten()(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # end: (None, classes)
    # print('end:', K.int_shape(x))
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model
def inception_resnet_v2(input_shape, classes): # input_shape=299 x 299 x 3
    x_input = Input(shape=input_shape)
    x = stem_v2(x_input) # (None, 35, 35, 384)
    for i in range(5):
        x = inception_resnet_a(x, f1=48, f2=64, f3=384) # (None, 35, 35, 384)
    x = reduction_a(x, f=384) # (None, 17, 17, 1152)
    for i in range(10):
        x = inception_resnet_b(x, f11=192, f21=160, f22=192, f2=1152) # (None, 17, 17, 1152)
    x = reduction_b(x, f1=288, f2=288, f3=320)  # (None, 8, 8, 2144)
    for i in range(5):
        x = inception_resnet_c(x, f1=224, f2=256, f3=2144) # (None, 8, 8, 2144)
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(x)  # (None, 1, 1, 2144)
    x = BatchNormalization(axis=-1)(x)
    x = Flatten()(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # end: (None, classes)
    # print('end:', K.int_shape(x))
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model
if __name__ == '__main__':
    inception_resnet_v1(input_shape=(299,299,3), classes=10)
    # inception_resnet_v2(input_shape=(299, 299, 3), classes=10)