import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# DenseNet
def ConvBlock(x, filters):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters * 4, use_bias=False,
               kernel_size=(1, 1), strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, use_bias=False,
               kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    return x

def TransitionBlock(x, filters, compression=1):
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = Conv2D(filters=int(filters * compression), use_bias=False,
               kernel_size=(1, 1), strides=(1, 1),
               padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x

def DenseBlock(x, layers, growth_rate):
    concat_feature = x
    for l in range(layers):
        x = ConvBlock(concat_feature, growth_rate)
        concat_feature = Concatenate(axis=-1)([concat_feature, x])
    return concat_feature

def densenet_model(x_shape, y_shape, use_bias=False, print_summary=False):
    _in = Input(shape=x_shape)
    x = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(_in)
    x = DenseBlock(x, 16, 12)
    x = TransitionBlock(x, x.shape[-1], 0.5)
    x = DenseBlock(x, 16, 12)
    x = TransitionBlock(x, x.shape[-1], 0.5)
    x = DenseBlock(x, 16, 12)
    x = GlobalAveragePooling2D()(x)
    _out = Dense(units=y_shape, use_bias=False, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name='DenseNet')
    if print_summary:
        model.summary()
    return model

# Smaller VGGNet
def vggnet_model(x_shape, y_shape, use_bias=False, print_summary=False):
    _in = Input(shape=x_shape)
    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(_in)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    # (CONV => RELU) * 2 => POOL
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # (CONV => RELU) * 2 => POOL
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # FC => RELU
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    _out = Dense(units=y_shape, use_bias=False, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name="SmallerVGGNet")
    if print_summary:
        model.summary()
    return model

# Resnet
def ConvBlock1(x):
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def ConvBlock2(x, num_blocks, filter_1, filter_2, first_strides):
    shortcut = x
    for i in range(num_blocks):
        if (i == 0):
            x = Conv2D(filter_1, (1, 1), strides=first_strides, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_1, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_2, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(filter_2, (1, 1), strides=first_strides, padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            shortcut = x
        else:
            x = Conv2D(filter_1, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_1, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filter_2, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            shortcut = x
    return x

def resnet_model(x_shape, y_shape, use_bias=False, print_summary=True):
    _in = Input(shape=x_shape)
    x = ConvBlock1(_in)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = ConvBlock2(x, num_blocks = 3, filter_1 = 32, filter_2 = 128, first_strides = (1, 1))
    x = ConvBlock2(x, num_blocks = 4, filter_1 = 64, filter_2 = 256, first_strides = (2, 2))
    x = ConvBlock2(x, num_blocks = 6, filter_1 = 128, filter_2 = 512, first_strides = (2, 2))
    x = ConvBlock2(x, num_blocks = 3, filter_1 = 256, filter_2 = 1024, first_strides = (2, 2))
    x = GlobalAveragePooling2D()(x)
    _out = Dense(units=y_shape, use_bias=False, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name='ResNet50')
    if print_summary:
        model.summary()
    return model