import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv3D, Conv2DTranspose,add,subtract,BatchNormalization
from keras.layers import Activation,Add,Flatten,Reshape,LeakyReLU,Concatenate,Lambda
from keras.layers.noise import GaussianNoise
from keras.applications import ResNet50
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
import tensorflow as tf
import time
import keras.backend as K


def batch_normal(x,use_nor=True,use_drop=False):
    if use_nor:
        x=BatchNormalization()(x)
    if use_drop:
        x=Dropout(0.5)(x)
    return x

def basic_block(x,filters,kernel_size,strides,padding,use_nor=False,use_drop=False,activation=None,is_activation=True):
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)(x)
    x=batch_normal(x,use_nor=use_nor,use_drop=use_drop)
    if is_activation:
        x=Activation(activation=activation)(x)
    return x

def residual_block(x,filters,kernel_size,strides,padding,activation,is_res_activation=False):
    use_drop=False
    use_nor=False

    y=basic_block(x,int(filters/4),(1,1),(1,1),padding,use_nor=use_nor,use_drop=use_drop,activation=activation,is_activation=False)
    y=basic_block(y,int(filters/4),(kernel_size,kernel_size),(strides,strides),padding,use_nor=use_nor,use_drop=use_drop,activation=activation,is_activation=False)
    y=basic_block(y,filters,(1,1),(1,1),padding,use_nor=use_nor,use_drop=use_drop,activation=activation,is_activation=False)

    if x.get_shape()[1:]!=y.get_shape()[1:]:
        x=basic_block(x,filters,(1,1),(strides,strides),padding,use_nor=use_nor,use_drop=use_drop,activation=None,is_activation=False)

    y=Add()([y,x])
    if is_res_activation:
        y=Activation(activation=activation)(y)
    return y

def build_model(input_img):

    activation='relu'
    kernel_size=4
    is_res_activation=True

    x=Conv2D(64,kernel_size=(4,4),strides=(1,1),padding='same',activation=activation)(input_img)
    x1=residual_block(x,  64,kernel_size=kernel_size,strides=2,padding='same',activation=activation,is_res_activation=is_res_activation)
    x2=residual_block(x1,128,kernel_size=kernel_size,strides=2,padding='same',activation=activation,is_res_activation=is_res_activation)
    x3=residual_block(x2,256,kernel_size=kernel_size,strides=2,padding='same',activation=activation,is_res_activation=is_res_activation)
    x4=residual_block(x3,512,kernel_size=kernel_size,strides=2,padding='same',activation=activation,is_res_activation=is_res_activation)
    x5=residual_block(x4,512,kernel_size=kernel_size,strides=1,padding='same',activation=activation,is_res_activation=is_res_activation)
    # x5=Conv2DTranspose(512,kernel_size=(4,4),strides=(1,1),padding='same')(x5)

    x5=Concatenate()([x5,x4])
    x6=Conv2DTranspose(256,kernel_size=(kernel_size,kernel_size),strides=(2,2),padding='same')(x5)
    x6=Activation(activation=activation)(x6)


    x6=Concatenate()([x6,x3])
    x7=Conv2DTranspose(128,kernel_size=(kernel_size,kernel_size),strides=(2,2),padding='same')(x6)
    x7=Activation(activation=activation)(x7)


    x7=Concatenate()([x7,x2])
    x8=Conv2DTranspose(64,kernel_size=(kernel_size,kernel_size),strides=(2,2),padding='same')(x7)
    x8=Activation(activation=activation)(x8)

    x8=Concatenate()([x8,x1])
    x9=Conv2DTranspose(64,kernel_size=(kernel_size,kernel_size),strides=(2,2),padding='same')(x8)
    x9=Activation(activation=activation)(x9)

    x10=Conv2D(3,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same',activation=None)(x9)


    model=Model(input_img,x10)

    #loss=keras.losses.logcosh

    # model.compile(optimizer=Adam(lr=0.001),loss='mae')

    return model

if __name__=='__main__':
    input_img=Input(shape=(224,224,3))
    # start=time.time()
    model=build_model(input_img)
    # end=time.time()
    # print(end-start)
    model.summary()
    # plot_model(model,'model.png',show_shapes=True)




