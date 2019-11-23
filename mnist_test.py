import numpy as np
import keras
from keras.models import Model,model_from_json,model_from_yaml,load_model,save_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical,plot_model
from keras.layers import Add,Conv2D,Dense,Dropout,Flatten,ZeroPadding2D,Activation,MaxPooling2D,AveragePooling2D,Input,LeakyReLU,BatchNormalization
import time
from PIL import Image
from keras import backend as K  # keras.backend.int_shape()
from resnet import easy_res_block,res_block
from inception_v1 import inception_module,my_conv,conv
from inception_v2 import inceptionv2_module

import os
os.environ["PATH"] += os.pathsep + 'D:\code\tensorflow-gpu\venv\Lib\site-packages\Graphviz2.38\bin'


(train_data, train_label), (test_data, test_label) = mnist.load_data()
train_data = train_data.reshape((60000, 28, 28, 1))
train_data = np.array(train_data) / 255

test_data = test_data.reshape((10000, 28, 28, 1))
test_data = np.array(test_data) / 255

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
# print(train_data.shape, test_data.shape)
# print(train_label.shape, test_label.shape)

def load_train_data(size=1024):
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape((60000, 28, 28, 1))
    train_data = np.array(train_data) / 255

    test_data = test_data.reshape((10000, 28, 28, 1))
    test_data = np.array(test_data) / 255

    train_label = to_categorical(train_label)
    test_label = to_categorical(test_label)
    n = 60000
    m=-1
    while True:
        m+=1
        m%=n//size+1    # 60000//1024+1=59
        yield train_data[m*size:min((m+1)*size,n)], train_label[m*size:min((m+1)*size,n)]
def load_test_data(size=1024):
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape((60000, 28, 28, 1))
    train_data = np.array(train_data) / 255

    test_data = test_data.reshape((10000, 28, 28, 1))
    test_data = np.array(test_data) / 255

    train_label = to_categorical(train_label)
    test_label = to_categorical(test_label)
    n = 10000
    m=-1
    while True:
        m+=1
        m%=n//size+1    # 10000//1024+1=10
        yield test_data[m*size:min((m+1)*size,n)], test_label[m*size:min((m+1)*size,n)]
def show():
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape((60000, 28, 28, 1))
    train_data = np.array(train_data)
    data = train_data[0:2]  # (2, 28, 28, 1)
    for i in data:
        data = i.reshape(28, 28)
        plt.imshow(data)
        plt.show()
def vgg16(input_shape, classes):     # 224 x 224 x 3
    x_input = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',      # 224 x 224 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',      # 224 x 224 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 112 x 112 x 64

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 112 x 112 x 128
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 112 x 112 x 128
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 56 x 56 x 128

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 56 x 56 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 56 x 56 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 56 x 56 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 28 x 28 x 256

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
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)          # 14 x 14 x 512

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
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)          # 7 x 7 x 512

    x = keras.layers.Flatten()(x)                                                   # 4096
    x = Dense(units=4096, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(units=4096, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(units=1000, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)

    model = keras.Model(inputs=x_input, outputs=x)
    return  model
def vgg19(input_shape, classes):    # 224 x 224 x 3    16 Conv layer and 3 Dense layer
    x_input = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',      # 224 x 224 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',      # 224 x 224 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 112 x 112 x 64

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 112 x 112 x 128
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 112 x 112 x 128
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 56 x 56 x 128

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 56 x 56 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 56 x 56 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 56 x 56 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',     # 56 x 56 x 256
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          # 28 x 28 x 256

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
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)          # 14 x 14 x 512

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
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)          # 7 x 7 x 512

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
def train_vgg(retrain=False, size=64, eps=5):
    if retrain:
        # model = keras.models.load_model('model/vgg16.h5')
        model = keras.models.load_model('model/vgg19.h5')
        fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        # model.save('model/vgg16.h5')
        model.save('model/vgg19.h5')
    else:
        # model = vgg16(input_shape=(28, 28, 1), classes=10)
        model = vgg19(input_shape=(28, 28, 1), classes=10)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        # model.save('model/vgg16.h5')
        model.save('model/vgg19.h5')

    # pred = model.evaluate(test_data, test_label, batch_size=512)
    # print(pred)
    loss=fit.history['loss']
    acc=fit.history['acc']
    val_acc=fit.history['val_acc']
    plt.plot(loss,color='g',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='r',label='val_acc')
    plt.title('vgg')
    plt.legend(loc='best')
    plt.show()
# def resnet18(input_shape, classes):
#     x_input = Input(shape=input_shape)  # 28 x 28 x 1
#     x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 28 x 28 x 64
#                kernel_regularizer=keras.regularizers.l2(1e-6),
#                kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_input)
#     x = BatchNormalization(axis=-1)(x)
#     x = LeakyReLU(alpha=0.05)(x)
#     x = solid_res_block(x, 64, 3, 1)    # 28 x 28 x 64
#     # print(keras.backend.int_shape(x))
#     x = solid_res_block(x, 64, 3, 1)  # 28 x 28 x 64
#     x = dotted_res_block(x, 128, 3, 2, 1)  # 14 x 14 x 128
#     x = solid_res_block(x, 128, 3, 1)   # 14 x 14 x 128
#     x = dotted_res_block(x, 256, 3, 2, 1)   # 7 x 7 x 256
#
#     x = solid_res_block(x, 256, 3, 1)   # 7 x 7 x 256
#     # print(K.int_shape(x))
#     # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
#     x = dotted_res_block(x, 512, 3, 2, 1)   # 4 x 4 x 512
#     # print(K.int_shape(x))
#
#     x = solid_res_block(x, 512, 3, 1)   # 4 x 4 x 512
#     x = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)  # 1 x 1 x 512
#     x = Flatten()(x)
#     x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
#     model = keras.models.Model(inputs=x_input, outputs=x)
#     return model
def resnet18(input_shape, classes): # input_shape=28 x 28 x 1
    x_input = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  # 28 x 28 x 64
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_input)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = easy_res_block(x, filte=64, size=3, stride=1, button=False) # (None, 28, 28, 64)
    x = easy_res_block(x, filte=64, size=3, stride=1, button=False) # (None, 28, 28, 64)
    x = easy_res_block(x, filte=128, size=3, stride=2, button=True) # (None, 14, 14, 128)
    x = easy_res_block(x, filte=128, size=3, stride=1, button=False) # (None, 14, 14, 128)
    x = easy_res_block(x, filte=256, size=3, stride=2, button=True)  # (None, 7, 7, 256)
    x = easy_res_block(x, filte=256, size=3, stride=1, button=False)  # (None, 7, 7, 256)
    x = easy_res_block(x, filte=512, size=3, stride=2, button=True)  # (None, 4, 4, 512)
    x = easy_res_block(x, filte=512, size=3, stride=1, button=False)  # (None, 4, 4, 512)
    x = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)  # (None, 1, 1, 512)
    # print(K.int_shape(x))
    x = Flatten()(x)    # (None, 512)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model
def train_resnet18(retrain=False, size=64, eps=5):
    if retrain:
        model = keras.models.load_model('model/resnet18.h5')
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000 // size, epochs=eps,
                                  validation_data=load_test_data(size=size), validation_steps=10000 // size)
        model.save('model/resnet18.h5')
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet18.json', 'w') as file:
            file.write(model_json)
        keras.utils.plot_model(model, to_file='draw/resnet18.png', show_shapes=True, show_layer_names=True)
    else:
        model = resnet18(input_shape=(28, 28, 1), classes=10)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000 // size, epochs=eps,
                                  validation_data=load_test_data(size=size), validation_steps=10000 // size)
        model.save('model/resnet18.h5')
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet18.json', 'w') as file:
            file.write(model_json)
        keras.utils.plot_model(model, to_file='draw/resnet18.png', show_shapes=True, show_layer_names=True)
    # pred = model.evaluate(test_data, test_label, batch_size=512)
    # print(pred)
    loss=fit.history['loss']
    acc=fit.history['acc']
    val_acc=fit.history['val_acc']
    plt.plot(loss,color='g',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='r',label='val_acc')
    plt.title('resnet18')
    plt.legend(loc='best')
    plt.show()
def resnet34(input_shape, classes): # input_shape=28 x 28 x 1
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # (None, 34, 34, 1)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',  # (None, 32, 32, 64)
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # (None, 16, 16, 64)
    for i in range(3):
        x = easy_res_block(x, filte=64, size=3, stride=1, button=False) # (None, 16, 16, 64)

    x = easy_res_block(x, filte=128, size=3, stride=2, button=True) # (None, 8, 8, 128)
    for i in range(3):
        x = easy_res_block(x, filte=128, size=3, stride=1, button=False)  # (None, 8, 8, 128)

    x = easy_res_block(x, filte=256, size=3, stride=2, button=True)  # (None, 4, 4, 256)
    for i in range(5):
        x = easy_res_block(x, filte=256, size=3, stride=1, button=False)  # (None, 4, 4, 256)

    x = easy_res_block(x, filte=512, size=3, stride=2, button=True)  # (None, 2, 2, 512)
    for i in range(2):
        x = easy_res_block(x, filte=512, size=3, stride=1, button=False)  # (None, 2, 2, 512)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # (None, 1, 1, 512)
    x = Flatten()(x)    # (None, 512)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print(keras.backend.int_shape(x))
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model
def train_resnet34(retrain=False, size=64, eps=5):
    if retrain:
        model = keras.models.load_model('model/resnet34.h5')
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000//size, epochs=eps,
                                  validation_data=load_test_data(size=size),validation_steps=10000//size)
        model.save('model/resnet34.h5')
        keras.utils.plot_model(model, to_file='draw/resnet34.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet34.json', 'w') as file:
            file.write(model_json)
    else:
        model = resnet34(input_shape=(28, 28, 1), classes=10)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000 // size, epochs=eps,
                                  validation_data=load_test_data(size=size), validation_steps=10000 // size)
        model.save('model/resnet34.h5')
        keras.utils.plot_model(model, to_file='draw/resnet34.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet34.json', 'w') as file:
            file.write(model_json)
    # pred = model.evaluate(test_data, test_label, batch_size=512)
    # pred = model.evaluate_generator(load_test_data(size=1024), steps=10000//1024)
    # print(pred)
    loss=fit.history['loss']
    acc=fit.history['acc']
    val_acc=fit.history['val_acc']
    plt.plot(loss,color='g',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='r',label='val_acc')
    plt.title('resnet34')
    plt.legend(loc='best')
    plt.show()
def resnet50(input_shape, classes):     # input_shape = 28 x 28 x 1
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # 34 x 34 x 1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', # (None, 32, 32, 64)
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # (None, 16, 16, 64)
    x = res_block(x, filte1=64, filte3=256, stride=1, button=True)      # (None, 16, 16, 256)
    for i in range(2):
        x = res_block(x, filte1=64, filte3=256, stride=1, button=False)  # (None, 16, 16, 256)

    x = res_block(x, filte1=128, filte3=512, stride=2, button=True)     # (None, 8, 8, 512)
    for i in range(3):
        x = res_block(x, filte1=128, filte3=512, stride=1, button=False)  # (None, 8, 8, 512)

    x = res_block(x, filte1=256, filte3=1024, stride=2, button=True)    # (None, 4, 4, 1024)
    for i in range(5):
        x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # (None, 4, 4, 1024)

    x = res_block(x, filte1=512, filte3=2048, stride=2, button=True)    # (None, 2, 2, 2048)
    for i in range(2):
        x = res_block(x, filte1=512, filte3=2048, stride=1, button=False)   # (None, 2, 2, 2048)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # 1 x 1 x 2048
    x = Flatten()(x)    # (None, 2048)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    # print(keras.backend.int_shape(x))
    model = keras.models.Model(inputs=x_input, outputs=x)
    return model
def train_resnet50(retrain=False, size=64, eps=5):
    if retrain:
        model = keras.models.load_model('model/resnet50.h5')
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000//size, epochs=eps,
                                  validation_data=load_test_data(size=size),validation_steps=10000//size)
        model.save('model/resnet50.h5')
        keras.utils.plot_model(model, to_file='draw/resnet50.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet50.json', 'w') as file:
            file.write(model_json)
    else:
        model = resnet50(input_shape=(28, 28, 1), classes=10)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000 // size, epochs=eps,
                                  validation_data=load_test_data(size=size), validation_steps=10000 // size)
        model.save('model/resnet50.h5')
        keras.utils.plot_model(model, to_file='draw/resnet50.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet50.json', 'w') as file:
            file.write(model_json)
    # pred = model.evaluate(test_data, test_label, batch_size=512)
    # pred = model.evaluate_generator(load_test_data(size=1024), steps=10000//1024)
    # print(pred)
    loss=fit.history['loss']
    acc=fit.history['acc']
    val_acc=fit.history['val_acc']
    plt.plot(loss,color='g',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='r',label='val_acc')
    plt.title('resnet50')
    plt.legend(loc='best')
    plt.show()
def resnet101(input_shape, classes):     # input_shape = 28 x 28 x 1
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # 34 x 34 x 1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', # (None, 34, 34, 64)
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)  # (None, 16, 16, 64)
    x = res_block(x, filte1=64, filte3=256, stride=1, button=True)      # (None, 16, 16, 256)
    for i in range(2):
        x = res_block(x, filte1=64, filte3=256, stride=1, button=False)  # (None, 16, 16, 256)

    x = res_block(x, filte1=128, filte3=512, stride=2, button=True)     # (None, 8, 8, 512)
    for i in range(3):
        x = res_block(x, filte1=128, filte3=512, stride=1, button=False)  # (None, 8, 8, 512)

    x = res_block(x, filte1=256, filte3=1024, stride=2, button=True)    # (None, 4 4, 1024)
    for i in range(22):
        x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # (None, 4, 4, 1024)

    x = res_block(x, filte1=512, filte3=2048, stride=2, button=True)    # (None, 2, 2, 2048)
    for i in range(2):
        x = res_block(x, filte1=512, filte3=2048, stride=1, button=False)   # (None, 2, 2, 2048)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # (None, 1, 1, 2048)
    x = Flatten()(x)    # (None, 2048)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print(keras.backend.int_shape(x))
    model = Model(inputs=x_input, outputs=x)
    return model
def train_resnet101(retrain=False, size=64, eps=5):
    if retrain:
        model = keras.models.load_model('model/resnet101.h5')
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000//size, epochs=eps,
                                  validation_data=load_test_data(size=size),validation_steps=10000//size)
        model.save('model/resnet101.h5')
        keras.utils.plot_model(model, to_file='draw/resnet101.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet101.json', 'w') as file:
            file.write(model_json)
    else:
        model = resnet101(input_shape=(28, 28, 1), classes=10)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000 // size, epochs=eps,
                                  validation_data=load_test_data(size=size), validation_steps=10000 // size)
        model.save('model/resnet101.h5')
        keras.utils.plot_model(model, to_file='draw/resnet101.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet101.json', 'w') as file:
            file.write(model_json)
    # pred = model.evaluate(test_data, test_label, batch_size=512)
    # pred = model.evaluate_generator(load_test_data(size=1024), steps=10000//1024)
    # print(pred)
    loss=fit.history['loss']
    acc=fit.history['acc']
    val_acc=fit.history['val_acc']
    plt.plot(loss,color='g',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='r',label='val_acc')
    plt.title('resnet101')
    plt.legend(loc='best')
    plt.show()
def resnet152(input_shape, classes):     # input_shape = 28 x 28 x 1
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # 34 x 34 x 1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', # (None, 34, 34, 64)
               kernel_regularizer=keras.regularizers.l2(1e-6),
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)  # (None, 16, 16, 64)
    x = res_block(x, filte1=64, filte3=256, stride=1, button=True)      # (None, 16, 16, 256)
    for i in range(2):
        x = res_block(x, filte1=64, filte3=256, stride=1, button=False)  # (None, 16, 16, 256)

    x = res_block(x, filte1=128, filte3=512, stride=2, button=True)     # (None, 8, 8, 512)
    for i in range(7):
        x = res_block(x, filte1=128, filte3=512, stride=1, button=False)  # (None, 8, 8, 512)

    x = res_block(x, filte1=256, filte3=1024, stride=2, button=True)    # (None, 4, 4, 1024)
    for i in range(35):
        x = res_block(x, filte1=256, filte3=1024, stride=1, button=False)   # (None, 4, 4, 1024)

    x = res_block(x, filte1=512, filte3=2048, stride=2, button=True)    # (None, 2, 2, 2048)
    for i in range(2):
        x = res_block(x, filte1=512, filte3=2048, stride=1, button=False)   # (None, 2, 2, 2048)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)  # (None, 1, 1, 2048)
    x = Flatten()(x)    # (None, 2048)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print(keras.backend.int_shape(x))
    model = Model(inputs=x_input, outputs=x)
    return model
def train_resnet152(retrain=False, size=64, eps=5):
    if retrain:
        model = keras.models.load_model('model/resnet152.h5')
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000//size, epochs=eps,
                                  validation_data=load_test_data(size=size),validation_steps=10000//size)
        model.save('model/resnet152.h5')
        keras.utils.plot_model(model, to_file='draw/resnet152.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet152.json', 'w') as file:
            file.write(model_json)
    else:
        model = resnet152(input_shape=(28, 28, 1), classes=10)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000 // size, epochs=eps,
                                  validation_data=load_test_data(size=size), validation_steps=10000 // size)
        model.save('model/resnet152.h5')
        keras.utils.plot_model(model, to_file='draw/resnet152.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/resnet152.json', 'w') as file:
            file.write(model_json)
    # pred = model.evaluate(test_data, test_label, batch_size=512)
    # pred = model.evaluate_generator(load_test_data(size=1024), steps=10000//1024)
    # print(pred)
    loss=fit.history['loss']
    acc=fit.history['acc']
    val_acc=fit.history['val_acc']
    plt.plot(loss,color='g',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='r',label='val_acc')
    plt.title('resnet152')
    plt.legend(loc='best')
    plt.show()
def googlenet(input_shape, classes):    # input_shape=28 x 28 x 1
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # (None, 34, 34, 1)
    # x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', # (None, 32, 32, 64)
    #                    kernel_regularizer=keras.regularizers.l2(1e-6),
    #                    kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    x = conv(x, filters=64, kernel_size=(3, 3), padding='valid')    # (None, 32, 32, 64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)   # (None, 16, 16, 64)
    x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',  # (None, 16, 16, 64)
    #            kernel_regularizer=keras.regularizers.l2(1e-6),
    #            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = my_conv(filters=64)(x)  # (None, 16, 16, 64)
    x = BatchNormalization(axis=-1)(x)
    # x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same',  # (None, 56, 56, 192)
    #            kernel_regularizer=keras.regularizers.l2(1e-6),
    #            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, 16, 16, 192)
    # x = BatchNormalization(axis=-1)(x)
    # x = LeakyReLU(alpha=0.05)(x)
    x = conv(x, filters=192, kernel_size=(3, 3))    # # (None, 16, 16, 192)
    x = inception_module(x, 64, 96, 128, 16, 32, 32, button=True)   # (None, 8, 8, 256)
    x = inception_module(x, 128, 128, 192, 32, 96, 64, button=False) # (None, 8, 8, 256)
    x = inception_module(x, 192, 96, 208, 16, 48, 64, button=True)  # (None, 4, 4, 512)
    x = inception_module(x, 160, 112, 224, 24, 64, 64, button=False)  # (None, 4, 4, 512)
    x = inception_module(x, 128, 128, 256, 24, 64, 64, button=False)  # (None, 4, 4, 512)
    x = inception_module(x, 112, 144, 288, 32, 64, 64, button=False)  # (None, 4, 4, 528)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, button=False)  # (None, 4, 4, 832)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, button=True)   # (None, 2, 2, 832)
    x = inception_module(x, 384, 192, 384, 48, 128, 128, button=False)  # (None, 2, 2, 1024)
    x = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)  # (None, 1, 1, 1024)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Flatten()(x)    # (None, 1024)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print('end:',K.int_shape(x))
    x = keras.models.Model(inputs=x_input, outputs=x)
    return x
def train_googlenet(retrain=False, size=64, eps=5):
    if retrain:
        model = keras.models.load_model('model/googlenet.h5')
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000//size, epochs=eps,
                                  validation_data=load_test_data(size=size),validation_steps=10000//size)
        model.save('model/googlenet.h5')
        keras.utils.plot_model(model, to_file='draw/googlenet.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/googlenet.json', 'w') as file:
            file.write(model_json)
    else:
        model = googlenet(input_shape=(28, 28, 1), classes=10)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000 // size, epochs=eps,
                                  validation_data=load_test_data(size=size), validation_steps=10000 // size)
        model.save('model/googlenet.h5')
        keras.utils.plot_model(model, to_file='draw/googlenet.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/googlenet.json', 'w') as file:
            file.write(model_json)
    # pred = model.evaluate(test_data, test_label, batch_size=512)
    # pred = model.evaluate_generator(load_test_data(size=1024), steps=10000//1024)
    # print(pred)
    loss=fit.history['loss']
    acc=fit.history['acc']
    val_acc=fit.history['val_acc']
    plt.plot(loss,color='g',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='r',label='val_acc')
    plt.title('googlenet')
    plt.legend(loc='best')
    plt.show()
def inception_v2(input_shape, classes): # input_shape=28 x 28 x 1
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # (None, 34, 34, 1)
    x = my_conv(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)    # (None, 32, 32, 64)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)   # (None, 16, 16, 64)
    x = my_conv(filters=64)(x)  # (None, 16, 16, 64)
    x = BatchNormalization(axis=-1)(x)
    x = conv(x, filters=192, kernel_size=(3, 3))    # (None, 16, 16, 192)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)   # (None, 8, 8, 192)
    x = inceptionv2_module(x, 64, 64, 64, 64, 96, 32, conv_button=True, avg_button=True, stride=1) # (None, 8, 8, 256)
    x = inceptionv2_module(x, 64, 64, 96, 64, 96, 64, conv_button=True, avg_button=True, stride=1) # (None, 8, 8, 320)
    x = inceptionv2_module(x, 0, 128, 160, 64, 96, 0, conv_button=False, avg_button=False, stride=2) # (None, 4, 4, 576)
    x = inceptionv2_module(x, 224, 64, 96, 96, 128, 128, conv_button=True, avg_button=True, stride=1) # (None, 4, 4, 576)
    x = inceptionv2_module(x, 192, 96, 128, 96, 128, 128, conv_button=True, avg_button=True, stride=1) # (None, 4, 4, 576)
    x = inceptionv2_module(x, 160, 128, 160, 128, 160, 96, conv_button=True, avg_button=True, stride=1) # (None, 4, 4, 576)
    x = inceptionv2_module(x, 96, 128, 192, 160, 192, 96, conv_button=True, avg_button=True, stride=1) # (None, 4, 4, 576)
    x = inceptionv2_module(x, 0, 128, 192, 192, 256, 0, conv_button=False, avg_button=False, stride=2) # (None, 2, 2, 1024)
    x = inceptionv2_module(x, 352, 192, 320, 160, 224, 128, conv_button=True, avg_button=True, stride=1) # (None, 2, 2, 1024)
    x = inceptionv2_module(x, 352, 192, 320, 192, 224, 128, conv_button=True, avg_button=True, stride=1) # (None, 2, 2, 1024)
    x = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)  # (None, 1, 1, 1024)
    x = conv(x, filters=1000)   # (None, 1, 1, 1000)
    x = Flatten()(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x) # (None, classes)
    # print('end:',K.int_shape(x))
    x = keras.models.Model(inputs=x_input, outputs=x)
    return x
def train_inception_v2(retrain=False, size=64, eps=5):
    if retrain:
        model = keras.models.load_model('model/inception_v2.h5')
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000//size, epochs=eps,
                                  validation_data=load_test_data(size=size),validation_steps=10000//size)
        model.save('model/inception_v2.h5')
        keras.utils.plot_model(model, to_file='draw/inception_v2.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/inception_v2.json', 'w') as file:
            file.write(model_json)
    else:
        model = inception_v2(input_shape=(28, 28, 1), classes=10)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # fit = model.fit(train_data, train_label, batch_size=size, epochs=eps, validation_data=(test_data, test_label))
        fit = model.fit_generator(load_train_data(size=size), steps_per_epoch=60000 // size, epochs=eps,
                                  validation_data=load_test_data(size=size), validation_steps=10000 // size)
        model.save('model/inception_v2.h5')
        keras.utils.plot_model(model, to_file='draw/inception_v2.png', show_shapes=True, show_layer_names=True)
        # 模型保存JSON文件
        model_json = model.to_json()
        with open(r'model/inception_v2.json', 'w') as file:
            file.write(model_json)
    # pred = model.evaluate(test_data, test_label, batch_size=512)
    # pred = model.evaluate_generator(load_test_data(size=1024), steps=10000//1024)
    # print(pred)
    loss=fit.history['loss']
    acc=fit.history['acc']
    val_acc=fit.history['val_acc']
    plt.plot(loss,color='g',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='r',label='val_acc')
    plt.title('inception_v2')
    plt.legend(loc='best')
    plt.show()


# if __name__=='__main__':
    # start=time.time()
    # train_vgg(retrain=True,size=256,eps=3)
    # train_resnet18(retrain=True, size=256, eps=1)
    # train_resnet34(retrain=True, size=256, eps=1)
    # train_resnet50(retrain=True, size=128, eps=1)
    # train_resnet101(retrain=True, size=64, eps=1)
    # train_resnet152(retrain=True, size=32, eps=1)
    # train_googlenet(retrain=True, size=512, eps=1)
    # train_inception_v2(retrain=True, size=512, eps=1)
    # end=time.time()
    # print('use time:')
    # print(end - start, str((end - start) / 60) + ' min')


    # for a,b in load_test_data(size=1024):
    #     print(len(a),len(b))

    # googlenet(input_shape=(28,28,1), classes=10)
    # inception_v2(input_shape=(28, 28, 1), classes=10)














