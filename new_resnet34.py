from keras.layers import Conv2D
from functools import wraps
import keras
from keras.layers import Activation,LeakyReLU,BatchNormalization
from keras.regularizers import l2,l1_l2
import keras.backend as K

@wraps(Conv2D)
def my_conv(*args,**kwargs):
    new_kwargs={'kernel_regularizer':l2(1e-7)}
    new_kwargs['padding']='same'
    new_kwargs['kernel_initializer']=keras.initializers.glorot_uniform(seed=0)
    new_kwargs.update(kwargs)
    return Conv2D(*args,**new_kwargs)
def conv(x,filters_my,kernel_size_my,strides_my,padding_my='same'):
    x_con=my_conv(filters=filters_my,kernel_size=kernel_size_my,strides=strides_my,padding=padding_my)(x)
    x_con=BatchNormalization(axis=-1)(x_con)
    x_con=LeakyReLU(alpha=0.02)(x_con)
    return x_con
def convolutional_blcok(x,filters,strides=(2,2),first=False):
    f1,f2,f3=filters
    block_all=[]
    x_short=x
    if first:
        strides=(1,1)

    x_short = conv(x_short, f3, kernel_size_my=(1, 1), strides_my=strides)
    block_all.append(x_short)
    x_new=conv(x,f1,kernel_size_my=(1,1),strides_my=strides)
    x_new = conv(x_new, f2, kernel_size_my=(3, 3), strides_my=(1,1))
    x_new = conv(x_new, f3, kernel_size_my=(1, 1), strides_my=(1,1))
    block_all.append(x_new)
    for i in range(30):
        x_new = conv(x, f1, kernel_size_my=(1, 1), strides_my=strides)
        x_new = conv(x_new, f2, kernel_size_my=(3, 3), strides_my=(1, 1))
        x_new = conv(x_new, f3, kernel_size_my=(1, 1), strides_my=(1, 1))
        block_all.append(x_new)
    x=keras.layers.Add()(block_all)
    return x
def ident_block(x,filters,strides=(1,1)):
    f1, f2, f3 = filters
    block_all = []
    x_short = x

    block_all.append(x_short)
    x_new = conv(x, f1, kernel_size_my=(1, 1), strides_my=strides)
    x_new = conv(x_new, f2, kernel_size_my=(3, 3), strides_my=(1, 1))
    x_new = conv(x_new, f3, kernel_size_my=(1, 1), strides_my=(1, 1))
    block_all.append(x_new)
    for i in range(30):
        x_new = conv(x, f1, kernel_size_my=(1, 1), strides_my=strides)
        x_new = conv(x_new, f2, kernel_size_my=(3, 3), strides_my=(1, 1))
        x_new = conv(x_new, f3, kernel_size_my=(1, 1), strides_my=(1, 1))
        block_all.append(x_new)
    x = keras.layers.Add()(block_all)
    return x
def resnet(input_shape=(32,32,3),classes=20):
    x_input=keras.layers.Input(input_shape)
    x=keras.layers.ZeroPadding2D((3,3))(x_input)
    x=conv(x,filters_my=64,kernel_size_my=(7,7),strides_my=(2,2),padding_my='valid')  #16

    x=convolutional_blcok(x,(4,4,256))      #8
    for i in range(2):
        x=ident_block(x,(4,4,256))

    x=convolutional_blcok(x,(8,8,512))  #4
    for i in range(3):
        x=ident_block(x,(8,8,512))

    x=convolutional_blcok(x,(16,16,1024))  #2
    for i in range(5):
        x=ident_block(x,(32,32,1024))
    x = keras.layers.AveragePooling2D()(x)
    x=keras.layers.Flatten()(x)
    x=keras.layers.Dense(classes,activation='softmax',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    model=keras.models.Model(inputs=x_input,outputs=x)
    return model


if __name__=='__main__':
    model=resnet()
    print(model.summary())
