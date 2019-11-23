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

def inceptionv2_module(x,f1, f21, f22, f31, f32, f4, conv_button=False, avg_button=False, stride=1):
    x_branch1 = x_branch2 = x_branch3 = x_branch4 = x
    # if conv_button:
    #     x_branch1 = my_conv(filters=f1)(x_branch1)
    #     x_branch1 = BatchNormalization(axis=-1)(x_branch1)
    #     # print('x_branch1:', K.int_shape(x_branch1))

    x_branch2 = conv(x_branch2, filters=f21)
    x_branch2 = my_conv(filters=f22, kernel_size=(3, 3), strides=(stride, stride))(x_branch2)
    x_branch2 = BatchNormalization(axis=-1)(x_branch2)
    # print('x_branch2:', K.int_shape(x_branch2))

    x_branch3 = conv(x_branch3, filters=f31)
    x_branch3 = conv(x_branch3, filters=f32, kernel_size=(3, 3))
    x_branch3 = my_conv(filters=f32, kernel_size=(3, 3), strides=(stride, stride))(x_branch3)
    x_branch3 = BatchNormalization(axis=-1)(x_branch3)
    # print('x_branch3:', K.int_shape(x_branch3))


    x_branch4 = AveragePooling2D(pool_size=(3, 3), strides=(stride, stride), padding='same')(x_branch4)
    # print('Max pool:', K.int_shape(x_branch4))
    if avg_button:
        x_branch4 = my_conv(filters=f4)(x_branch4)
        x_branch4 = BatchNormalization(axis=-1)(x_branch4)
        # print('x_branch4:', K.int_shape(x_branch4))

    if conv_button:
        x_branch1 = my_conv(filters=f1)(x_branch1)
        x_branch1 = BatchNormalization(axis=-1)(x_branch1)
        x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3, x_branch4])
    else:
        x = keras.layers.concatenate([x_branch2, x_branch3, x_branch4])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def inception_v2(input_shape, classes): # input_shape=224 x 224 x 3
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # (None, 230, 230, 3)
    x = my_conv(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(x)    # (None, 112, 112, 64)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)   # (None, 56, 56, 64)
    x = my_conv(filters=64)(x)  # (None, 56, 56, 64)
    x = BatchNormalization(axis=-1)(x)
    x = conv(x, filters=192, kernel_size=(3, 3))    # (None, 56, 56, 192)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)   # (None, 28, 28, 192)
    x = inceptionv2_module(x, 64, 64, 64, 64, 96, 32, conv_button=True, avg_button=True, stride=1) # (None, 28, 28, 256)
    x = inceptionv2_module(x, 64, 64, 96, 64, 96, 64, conv_button=True, avg_button=True, stride=1) # (None, 28, 28, 320)
    x = inceptionv2_module(x, 0, 128, 160, 64, 96, 0, conv_button=False, avg_button=False, stride=2) # (None, 14, 14, 576)
    x = inceptionv2_module(x, 224, 64, 96, 96, 128, 128, conv_button=True, avg_button=True, stride=1) # (None, 14, 14, 576)
    x = inceptionv2_module(x, 192, 96, 128, 96, 128, 128, conv_button=True, avg_button=True, stride=1) # (None, 14, 14, 576)
    x = inceptionv2_module(x, 160, 128, 160, 128, 160, 96, conv_button=True, avg_button=True, stride=1) # (None, 14, 14, 576)
    x = inceptionv2_module(x, 96, 128, 192, 160, 192, 96, conv_button=True, avg_button=True, stride=1) # (None, 14, 14, 576)
    x = inceptionv2_module(x, 0, 128, 192, 192, 256, 0, conv_button=False, avg_button=False, stride=2) # (None, 7, 7, 1024)
    x = inceptionv2_module(x, 352, 192, 320, 160, 224, 128, conv_button=True, avg_button=True, stride=1) # (None, 7, 7, 1024)
    x = inceptionv2_module(x, 352, 192, 320, 192, 224, 128, conv_button=True, avg_button=True, stride=1) # (None, 7, 7, 1024)
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(x)  # (None, 1, 1, 1024)
    x = conv(x, filters=1000)   # (None, 1, 1, 1000)
    x = Flatten()(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print('end:',K.int_shape(x))
    x = keras.models.Model(inputs=x_input, outputs=x)
    return x

if __name__ == '__main__':
    inception_v2(input_shape=(224,224,3), classes=10)