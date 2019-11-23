import numpy as np
import keras
from keras.layers import Add,Conv2D,Dense,Dropout,Flatten,AveragePooling2D,MaxPooling2D,Input,LeakyReLU,BatchNormalization,ZeroPadding2D
from keras.regularizers import l2
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
def block(x, f1, f2, f3, stride=(1, 1), button=False):
    x_branch1 = conv(x, filters=f1)
    x_branch1 = conv(x_branch1, filters=f1, kernel_size=(3, 3), strides=stride)
    x_branch1 = conv(x_branch1, filters=f2)
    x_branch1_left = x_branch1 # (None, 56, 56, 256)
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=f3)
    x_branch1_right = conv(x_branch1_right, filters=f2) # (None, 56, 56, 256)
    if button:
        x_branch2 = conv(x, filters=f2, strides=stride) # (None, 56, 56, 256)
        x = keras.layers.Add()([x_branch1_left, x_branch1_right, x_branch2])
    else:
        x = keras.layers.Add()([x_branch1_left, x_branch1_right, x])
    x = conv(x, filters=f2)
    return x
'''
def block_a1(x, f1, f2, f3, stride=(1, 1)):
    x_branch1 = conv(x, filters=f1)
    x_branch1 = conv(x_branch1, filters=f1, kernel_size=(3, 3), strides=stride)
    x_branch1 = conv(x_branch1, filters=f2)
    x_branch1_left = x_branch1 # (None, 56, 56, 256)
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=f3)
    x_branch1_right = conv(x_branch1_right, filters=256) # (None, 56, 56, 256)
    x_branch2 = conv(x, filters=256, strides=stride) # (None, 56, 56, 256)
    x = keras.layers.Add()([x_branch1_left, x_branch1_right, x_branch2])
    x = conv(x, filters=f2)
    return x
def block_b1(x, f1, f2, f3, stride=(1, 1)):
    x_branch1 = conv(x, filters=f1)
    x_branch1 = conv(x_branch1, filters=f1, kernel_size=(3, 3))
    x_branch1 = conv(x_branch1, filters=f2)
    x_branch1_left = x_branch1  # (None, 56, 56, 256)
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=f3)
    x_branch1_right = conv(x_branch1_right, filters=f2)  # (None, 56, 56, 256)
    x = keras.layers.Add()([x_branch1_left, x_branch1_right, x])
    x = conv(x, filters=f2)
    return x
def block_a2(x):
    x_branch1 = conv(x, filters=256)
    x_branch1 = conv(x_branch1, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x_branch1 = conv(x_branch1, filters=512)
    x_branch1_left = x_branch1 # (None, 28, 28, 512)
    # print(K.int_shape(x_branch1_left))
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=32)
    x_branch1_right = conv(x_branch1_right, filters=512) # (None, 28, 28, 512)
    # print(K.int_shape(x_branch1_right))

    x_branch2 = conv(x, filters=512, strides=(2, 2), padding='valid') # (None, 28, 28, 512)
    # print(K.int_shape(x_branch2))
    x = keras.layers.Add()([x_branch1_left, x_branch1_right, x_branch2])

    x = conv(x, filters=512)
    return x
def block_b2(x):
    x_branch1 = conv(x, filters=256)
    x_branch1 = conv(x_branch1, filters=256, kernel_size=(3, 3))
    x_branch1 = conv(x_branch1, filters=512)
    x_branch1_left = x_branch1  # (None, 28, 28, 512)
    # print(K.int_shape(x_branch1_left))
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=32)
    x_branch1_right = conv(x_branch1_right, filters=512)  # (None, 28, 28, 512)
    # print(K.int_shape(x_branch1_right))
    x = keras.layers.Add()([x_branch1_left, x_branch1_right, x])
    x = conv(x, filters=512)
    return x
def block_a3(x):
    x_branch1 = conv(x, filters=512)
    x_branch1 = conv(x_branch1, filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x_branch1 = conv(x_branch1, filters=1024)
    x_branch1_left = x_branch1 # (None, 14, 14, 1024)
    # print(K.int_shape(x_branch1_left))
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=64)
    x_branch1_right = conv(x_branch1_right, filters=1024) # (None, 14, 14, 1024)
    # print(K.int_shape(x_branch1_right))
    x_branch2 = conv(x, filters=1024, strides=(2, 2), padding='valid') # (None, 14, 14, 1024)
    # print(K.int_shape(x_branch2))
    x = keras.layers.Add()([x_branch1_left, x_branch1_right, x_branch2])
    x = conv(x, filters=1024)
    return x
def block_b3(x):
    x_branch1 = conv(x, filters=512)
    x_branch1 = conv(x_branch1, filters=512, kernel_size=(3, 3))
    x_branch1 = conv(x_branch1, filters=1024)
    x_branch1_left = x_branch1  # (None, 28, 28, 512)
    # print(K.int_shape(x_branch1_left))
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=64)
    x_branch1_right = conv(x_branch1_right, filters=1024)  # (None, 28, 28, 512)
    # print(K.int_shape(x_branch1_right))
    x = keras.layers.Add()([x_branch1_left, x_branch1_right, x])
    x = conv(x, filters=1024)
    return x
def block_a4(x):
    x_branch1 = conv(x, filters=1024)
    x_branch1 = conv(x_branch1, filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x_branch1 = conv(x_branch1, filters=2048)
    x_branch1_left = x_branch1 # (None, 7, 7, 2048)
    # print(K.int_shape(x_branch1_left))
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=128)
    x_branch1_right = conv(x_branch1_right, filters=2048) # (None, 7, 7, 2048)
    # print(K.int_shape(x_branch1_right))
    x_branch2 = conv(x, filters=2048, strides=(2, 2), padding='valid') # (None, 7, 7, 2048)
    # print(K.int_shape(x_branch2))
    x = keras.layers.Add()([x_branch1_left, x_branch1_right, x_branch2])
    x = conv(x, filters=2048)
    return x
def block_b4(x):
    x_branch1 = conv(x, filters=1024)
    x_branch1 = conv(x_branch1, filters=1024, kernel_size=(3, 3))
    x_branch1 = conv(x_branch1, filters=2048)
    x_branch1_left = x_branch1  # (None, 7, 7, 2048)
    # print(K.int_shape(x_branch1_left))
    x_branch1_right = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_branch1)
    x_branch1_right = conv(x_branch1_right, filters=128)
    x_branch1_right = conv(x_branch1_right, filters=2048)  # (None, 7, 7, 2048)
    # print(K.int_shape(x_branch1_right))
    x = keras.layers.Add()([x_branch1_left, x_branch1_right, x])
    x = conv(x, filters=2048)
    return x
'''
def se_resnext_50(input_shape, classes):    # x_input=224 x 224 x 3
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input) #( None, 230, 230, 3)
    x = conv(x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid') # (None, 112, 112, 64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x) # (None, 55, 55, 64)
    x = block(x, 128, 256, 16, stride=(1, 1), button=True) # (None, 56, 56, 256)
    for i in range(2):
        x = block(x, 128, 256, 16, stride=(1, 1), button=False) # (None, 56, 56, 256)
    x = block(x, 256, 512, 32, stride=(2, 2), button=True) # (None, 28, 28, 512)
    for i in range(3):
        x = block(x, 256, 512, 32, stride=(1, 1), button=False) # (None, 28, 28, 512)
    x = block(x, 512, 1024, 64, stride=(2, 2), button=True) # (None, 14, 14, 1024)
    for i in range(5):
        x = block(x, 512, 1024, 64, stride=(1, 1), button=False) # (None, 14, 14, 1024)
    x = block(x, 1024, 2048, 128, stride=(2, 2), button=True) # (None, 7, 7, 2048)
    for i in range(2):
        x = block(x, 1024, 2048, 128, stride=(1, 1), button=False) # (None, 7, 7, 2048)
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(x) # (None, 1, 1, 2048)
    x = BatchNormalization(axis=-1)(x)
    x = Flatten()(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)  # end: (None, classes)
    # print('end:', K.int_shape(x))
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model

if __name__ == '__main__':
    se_resnext_50(input_shape=(224, 224, 3), classes=20)