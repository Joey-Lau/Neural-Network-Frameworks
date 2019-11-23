import numpy as np
import keras
from keras.layers import Add,Conv2D,Dense,Dropout,Flatten,Activation,AveragePooling2D,MaxPooling2D,Input,LeakyReLU,BatchNormalization,ZeroPadding2D
from keras.models import Model,Sequential
from keras.regularizers import l2
from keras.initializers import glorot_uniform
# from keras.applications.mobilenet import relu6,DepthwiseConv2D
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
from functools import wraps
import time

'''
def solid_res_block(x, filter, size, stride):
    x_branch = x
    x = Conv2D(filters=filter, kernel_size=(size, size), strides=(stride, stride), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2D(filters=filter, kernel_size=(size, size), strides=(stride, stride), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)

    x = keras.layers.Add()([x, x_branch])
    x = LeakyReLU(alpha=0.05)(x)
    return x
def dotted_res_block(x, filter, size, stride1, stride2):
    x_branch = x
    x = Conv2D(filters=filter, kernel_size=(size, size), strides=(stride1, stride1), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2D(filters=filter, kernel_size=(size, size), strides=(stride2, stride2), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)    # 16 x 16 x 128

    x_branch = Conv2D(filters=filter, kernel_size=(size, size), strides=(stride1, stride1), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_branch)
    x_branch = BatchNormalization(axis=-1)(x_branch)    # 16 x 16 x 128

    x = keras.layers.Add()([x, x_branch])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def resnet18(input_shape, classes):
    x_input = Input(shape=input_shape)  # 32 x 32 x 3
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 32 x 32 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_input)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)

    # x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 32 x 32 x 64
    #            kernel_regularizer=keras.regularizers.l2(1e-6),
    #            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_branch)
    # x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    #
    # x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 32 x 32 x 64
    #            kernel_regularizer=keras.regularizers.l2(1e-6),
    #            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    #
    # x_branch = keras.layers.Add()([x, x_branch])    # 32 x 32 x 64
    x = solid_res_block(x, 64, 3, 1)    # 32 x 32 x 64
    # print(K.int_shape(x))

    # x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 32 x 32 x 64
    #            kernel_regularizer=keras.regularizers.l2(1e-6),
    #            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_branch)
    # x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    # x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 32 x 32 x 64
    #            kernel_regularizer=keras.regularizers.l2(1e-6),
    #            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    x = solid_res_block(x, 64, 3, 1)  # 32 x 32 x 64
    # print(K.int_shape(x))

    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)    # 16 x 16 x 64
    # print(K.int_shape(x))

    # x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 16 x 16 x 128
    #            kernel_regularizer=keras.regularizers.l2(1e-6),
    #            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    # x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 16 x 16 x 128
    #            kernel_regularizer=keras.regularizers.l2(1e-6),
    #            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    x = dotted_res_block(x, 128, 3, 2, 1)  # 16 x 16 x 128
    # print(K.int_shape(x))

    x = solid_res_block(x, 128, 3, 1)   # 16 x 16 x 128
    # print(K.int_shape(x))

    x = dotted_res_block(x, 256, 3, 2, 1)   # 8 x 8 x 256
    # print(K.int_shape(x))

    x = solid_res_block(x, 256, 3, 1)   # 8 x 8 x 256

    x = dotted_res_block(x, 512, 3, 2, 1)   # 4 x 4 x 512
    # print(K.int_shape(x))

    x = solid_res_block(x, 512, 3, 1)   # 4 x 4 x 512

    x = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)  # 1 x 1 x 512
    # print(K.int_shape(x))
    x = Flatten()(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model
'''

def easy_res_block(x, filte, size, stride, button=False):
    # button=False is solid line, button=True is dotted line.
    # If stride=2, button=True
    x_branch = x    # 32 32 64

    x = Conv2D(filters=filte, kernel_size=(size, size), strides=(stride, stride), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)

    x = Conv2D(filters=filte, kernel_size=(size, size), strides=(1, 1), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)    # 16 16 128

    if button:
        x_branch = Conv2D(filters=filte, kernel_size=(size, size), strides=(stride, stride), padding='same',
                   kernel_regularizer=keras.regularizers.l2(1e-6),
                   kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_branch)
        x_branch = BatchNormalization(axis=-1)(x_branch)

    x = keras.layers.Add()([x, x_branch])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def resnet18(input_shape, classes): # input_shape=32 x 32 x 3
    x_input = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 32 x 32 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_input)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = easy_res_block(x, filte=64, size=3, stride=1, button=False) # (None, 32, 32, 64)
    x = easy_res_block(x, filte=64, size=3, stride=1, button=False) # (None, 32, 32, 64)
    x = easy_res_block(x, filte=128, size=3, stride=2, button=True) # (None, 16, 16, 128)
    x = easy_res_block(x, filte=128, size=3, stride=1, button=False) # (None, 16, 16, 128)
    x = easy_res_block(x, filte=256, size=3, stride=2, button=True)  # (None, 8, 8, 256)
    x = easy_res_block(x, filte=256, size=3, stride=1, button=False)  # (None, 8, 8, 256)
    x = easy_res_block(x, filte=512, size=3, stride=2, button=True)  # (None, 4, 4, 512)
    x = easy_res_block(x, filte=512, size=3, stride=1, button=False)  # (None, 4, 4, 512)
    x = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)  # (None, 1, 1, 512)
    # print(K.int_shape(x))
    x = Flatten()(x)    # (None, 512)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model

def resnet34(input_shape, classes): # input_shape=224 x 224 x 3
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # 230 x 230 x 3
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',  # (None, 115, 115, 64)
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # (None, 57, 57, 64)
    for i in range(3):
        x = easy_res_block(x, filte=64, size=3, stride=1, button=False) # (None, 57, 57, 64)

    x = easy_res_block(x, filte=128, size=3, stride=2, button=True) # (None, 29, 29, 128)
    for i in range(3):
        x = easy_res_block(x, filte=128, size=3, stride=1, button=False)  # (None, 29, 29, 128)

    x = easy_res_block(x, filte=256, size=3, stride=2, button=True)  # (None, 15, 15, 256)
    for i in range(5):
        x = easy_res_block(x, filte=256, size=3, stride=1, button=False)  # (None, 15, 15, 256)

    x = easy_res_block(x, filte=512, size=3, stride=2, button=True)  # (None, 8, 8, 512)
    for i in range(2):
        x = easy_res_block(x, filte=512, size=3, stride=1, button=False)  # (None, 8, 8, 512)

    x = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)  # (None, 2, 2, 512)
    x = Flatten()(x)    # (None, 2048)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print(keras.backend.int_shape(x))
    model = Model(inputs=x_input, outputs=x)
    return model

def res_block(x, filte1, filte3,stride=1, button=False):
    # button=False is solid line, button=True is dotted line.
    # If stride=2, button=True
    x_branch = x
    x = Conv2D(filters=filte1, kernel_size=(1, 1), strides=(stride, stride), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2D(filters=filte1, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2D(filters=filte3, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)

    if button:
        x_branch = Conv2D(filters=filte3, kernel_size=(3, 3), strides=(stride, stride), padding='same',
                          kernel_regularizer=keras.regularizers.l2(1e-6),
                          kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_branch)
        x_branch = BatchNormalization(axis=-1)(x_branch)

    x = keras.layers.Add()([x, x_branch])
    x = LeakyReLU(alpha=0.05)(x)

    return x

def resnet50(input_shape, classes):     # input_shape = 224 x 224 x 3
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # 230 x 230 x 3
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', # 112 x 112 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56 x 56 x 64
    x = res_block(x, filte1=64, filte3=256, stride=1, button=True)      # 56 x 56 x 256
    for i in range(2):
        x = res_block(x, filte1=64, filte3=256, stride=1, button=False)  # 56 x 56 x 256
    # x = res_block(x, filte1=64, filte3=256, stride=1, button=False)     # 56 x 56 x 256
    # x = res_block(x, filte1=64, filte3=256, stride=1, button=False)     # 56 x 56 x 256

    x = res_block(x, filte1=128, filte3=512, stride=2, button=True)     # 28 x 28 x 512
    for i in range(3):
        x = res_block(x, filte1=128, filte3=512, stride=1, button=False)  # 28 x 28 x 512
    # x = res_block(x, filte1=128, filte3=512, stride=1, button=False)    # 28 x 28 x 512
    # x = res_block(x, filte1=128, filte3=512, stride=1, button=False)    # 28 x 28 x 512
    # x = res_block(x, filte1=128, filte3=512, stride=1, button=False)    # 28 x 28 x 512

    x = res_block(x, filte1=256, filte3=1024, stride=2, button=True)    # 14 x 14 x 1024
    for i in range(5):
        x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # 14 x 14 x 1024
    # x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # 14 x 14 x 1024
    # x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # 14 x 14 x 1024
    # x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # 14 x 14 x 1024
    # x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # 14 x 14 x 1024
    # x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # 14 x 14 x 1024

    x = res_block(x, filte1=512, filte3=2048, stride=2, button=True)    # 7 x 7 x 2048
    for i in range(2):
        x = res_block(x, filte1=512, filte3=2048, stride=1, button=False)   # 7 x 7 x 2048
    # x = res_block(x, filte1=512, filte3=2048, stride=1, button=False)   # 7 x 7 x 2048
    # x = res_block(x, filte1=512, filte3=2048, stride=1, button=False)   # 7 x 7 x 2048

    x = AveragePooling2D(pool_size=(7, 7), strides=(2, 2), padding='valid')(x)  # 1 x 1 x 2048
    x = Flatten()(x)    # (None, 2048)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print(keras.backend.int_shape(x))
    model = Model(inputs=x_input, outputs=x)
    return model

def resnet101(input_shape, classes):     # input_shape = 224 x 224 x 3
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # 230 x 230 x 3
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', # (None, 112, 112, 64)
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # (None, 56, 56, 64)
    x = res_block(x, filte1=64, filte3=256, stride=1, button=True)      # (None, 56, 56, 256)
    for i in range(2):
        x = res_block(x, filte1=64, filte3=256, stride=1, button=False)  # (None, 56, 56, 256)

    x = res_block(x, filte1=128, filte3=512, stride=2, button=True)     # (None, 28, 28, 512)
    for i in range(3):
        x = res_block(x, filte1=128, filte3=512, stride=1, button=False)  # (None, 28, 28, 512)

    x = res_block(x, filte1=256, filte3=1024, stride=2, button=True)    # (None, 14, 14, 1024)
    for i in range(22):
        x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # (None, 14, 14, 1024)

    x = res_block(x, filte1=512, filte3=2048, stride=2, button=True)    # (None, 7, 7, 2048)
    for i in range(2):
        x = res_block(x, filte1=512, filte3=2048, stride=1, button=False)   # (None, 7, 7, 2048)

    x = AveragePooling2D(pool_size=(7, 7), strides=(2, 2), padding='valid')(x)  # (None, 1, 1, 2048)
    x = Flatten()(x)    # (None, 2048)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print(keras.backend.int_shape(x))
    model = Model(inputs=x_input, outputs=x)
    return model

def resnet152(input_shape, classes):     # input_shape = 224 x 224 x 3
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # 230 x 230 x 3
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', # (None, 112, 112, 64)
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # (None, 56, 56, 64)
    x = res_block(x, filte1=64, filte3=256, stride=1, button=True)      # (None, 56, 56, 64)
    for i in range(2):
        x = res_block(x, filte1=64, filte3=256, stride=1, button=False)  # (None, 56, 56, 64)

    x = res_block(x, filte1=128, filte3=512, stride=2, button=True)     # (None, 28, 28, 512)
    for i in range(7):
        x = res_block(x, filte1=128, filte3=512, stride=1, button=False)  # ((None, 28, 28, 512)

    x = res_block(x, filte1=256, filte3=1024, stride=2, button=True)    # (None, 14, 14, 1024)
    for i in range(35):
        x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # (None, 14, 14, 1024)

    x = res_block(x, filte1=512, filte3=2048, stride=2, button=True)    # (None, 7, 7, 2048)
    for i in range(2):
        x = res_block(x, filte1=512, filte3=2048, stride=1, button=False)   # (None, 7, 7, 2048)

    x = AveragePooling2D(pool_size=(7, 7), strides=(2, 2), padding='valid')(x)  # (None, 1, 1, 2048)
    x = Flatten()(x)    # (None, 2048)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print(keras.backend.int_shape(x))
    model = Model(inputs=x_input, outputs=x)
    return model

if __name__ == '__main__':
    # resnet18(input_shape=(32, 32, 3), classes=10)
    # resnet34(input_shape=(224, 224, 3), classes=10)
    # resnet50(input_shape=(224, 224, 3), classes=10)
    # resnet101(input_shape=(224, 224, 3), classes=10)
    resnet152(input_shape=(224, 224, 3), classes=10)