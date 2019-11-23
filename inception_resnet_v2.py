from keras.layers import Conv2D
from functools import wraps
import keras
from keras import layers
from keras.layers import Activation,LeakyReLU,BatchNormalization,MaxPooling2D
from keras.regularizers import l2
from keras.models import Model

@wraps(Conv2D)
def my_conv(*args,**kwargs):
    new_kwargs={'kernel_regularizer':l2(5e-5)}
    new_kwargs['padding']='same'
    new_kwargs['strides']=(2,2) if kwargs.get('strides')==(2,2) else (1,1)
    # new_kwargs['kernel_initializer']=keras.initializers.glorot_uniform(seed=0)
    new_kwargs.update(kwargs)
    return Conv2D(*args,**new_kwargs)
def conv(x,**kwargs):
    x=my_conv(**kwargs)(x)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)
    return x
def stem_block(x):
    # x_tet=keras.Input(x_input)
    # x=my_conv(filters=32,kernel_size=(3,3))(x)  #,strides=(2,2)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x=conv(x,filters=32,kernel_size=(3,3))
    x = conv(x, filters=64, kernel_size=(3, 3))

    s1=MaxPooling2D((3,3),strides=(2,2),padding='same')(x)
    s2=conv(x,filters=96,kernel_size=(3,3),strides=(2,2))

    x=keras.layers.concatenate([s1,s2])

    s1 =conv(x,filters=64, kernel_size=(1, 1))
    s1=conv(s1,filters=96,kernel_size=(3,3))
    s2=conv(x,filters=64,kernel_size=(1,1))
    s2=conv(s2,filters=64,kernel_size=(7,1))
    s2=conv(s2,filters=64,kernel_size=(1,7))
    s2=conv(s2,filters=96,kernel_size=(3,3))

    x=keras.layers.concatenate([s1,s2])

    s1=conv(x,filters=192,kernel_size=(3,3))      #,strides=(2,2)
    s2=MaxPooling2D((1,1),padding='same')(x)     #pool_size??????????   ,strides=(2,2)
    x=keras.layers.concatenate([s1,s2])
    x=LeakyReLU(alpha=0.05)(x)
    # model=Model(inputs=x_tet,outputs=x)
    # return model
    return x
def inception_resnet_a(x_input):
    x_short=x_input
    s1=conv(x_input,filters=32,kernel_size=(1,1))

    s2=conv(x_input,filters=32,kernel_size=(1,1))
    s2=conv(s2,filters=32,kernel_size=(3,3))

    s3=conv(x_input,filters=32,kernel_size=(1,1))
    s3=conv(s3,filters=48,kernel_size=(3,3))
    s3=conv(s3,filters=64,kernel_size=(3,3))
    x=keras.layers.concatenate([s1,s2,s3])
    x=conv(x,filters=384,kernel_size=(1,1))
    x=layers.Add()([x_short,x])
    x=LeakyReLU(alpha=0.05)(x)

    # model=Model(inputs=x_input,outputs=x)
    # return model
    return x
def reduction_a(x_input,stride=(2,2)):
    s1=MaxPooling2D((3,3),strides=stride,padding='same')(x_input)

    s2=conv(x_input,filters=384,strides=stride,kernel_size=(3,3))

    s3=conv(x_input,filters=256,kernel_size=(1,1))
    s3=conv(s3,filters=256,kernel_size=(3,3))
    s3=conv(s3,filters=384,kernel_size=(3,3),strides=stride)

    x=layers.concatenate([s1,s2,s3])
    x = LeakyReLU(alpha=0.05)(x)
    return x
    # model=Model(inputs=x_input,outputs=x)
    # return model
def inception_resnet_b(x_input):
    x_short=x_input
    s1=conv(x_input,filters=192,kernel_size=(1,1))

    s2=conv(x_input,filters=128,kernel_size=(1,1))
    s2=conv(s2,filters=160,kernel_size=(1,7))
    s2=conv(s2,filters=192,kernel_size=(7,1))
    x=layers.Add()([s1,s2])
    x=conv(x,filters=1152,kernel_size=(1,1))
    x=layers.Add()([x_short,x])
    # x=layers.concatenate([x_short,x])
    x=LeakyReLU(alpha=0.05)(x)
    # model = Model(inputs=x_input, outputs=x)
    # return model
    return x
def reduction_b(x_input,stride=(2,2)):
    s1=MaxPooling2D((3,3),strides=stride,padding='same')(x_input)

    s2=conv(x_input,filters=256,kernel_size=(1,1))
    s2=conv(s2,filters=384,kernel_size=(3,3),strides=stride)

    s3 = conv(x_input, filters=256, kernel_size=(1, 1))
    s3 = conv(s3, filters=288, kernel_size=(3, 3), strides=stride)

    s4 = conv(x_input, filters=256, kernel_size=(1, 1))
    s4 = conv(s4, filters=288, kernel_size=(3, 3))
    s4 = conv(s4, filters=320, kernel_size=(3, 3), strides=stride)

    x=layers.concatenate([s1,s2,s3,s4])
    x = LeakyReLU(alpha=0.05)(x)
    # model = Model(inputs=x_input, outputs=x)
    # return model
    return x
def inception_resnet_c(x_input):
    x_short=x_input

    s1=conv(x_input,filters=192,kernel_size=(1,1))

    s2=conv(x_input,filters=192,kernel_size=(1,1))
    s2=conv(s2,filters=224,kernel_size=(1,3))
    s2=conv(s2,filters=256,kernel_size=(3,1))
    x=layers.concatenate([s1,s2])

    x=conv(x,filters=2144,kernel_size=(1,1))
    x=layers.Add()([x_short,x])
    x=LeakyReLU(alpha=0.05)(x)
    # model = Model(inputs=x_input, outputs=x)
    # return model
    return x
def inception_resnet_v2_model(input_shape,classes):
    x_input=keras.layers.Input(input_shape)
    # x=my_conv(filters=64,kernel_size=(1,1))(x_input)
    x = my_conv(filters=32, kernel_size=(3, 3))(x_input)
    x=stem_block(x)

    for i in range(2):
        x=inception_resnet_a(x)

    x=reduction_a(x,stride=(2,2))

    for i in range(4):
        x=inception_resnet_b(x)

    x=reduction_b(x,stride=(2,2))

    for i in range(2):
        x=inception_resnet_c(x)

    x=layers.AveragePooling2D(pool_size=(4,4),strides=(1,1))(x)
    x = keras.layers.Flatten()(x)
    x=layers.Dropout(rate=0.2)(x)
    x=layers.Dense(classes,activation='softmax',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    model=Model(inputs=x_input,outputs=x)
    return model