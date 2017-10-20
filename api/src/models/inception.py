from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.utils.data_utils import get_file

from ..common.config import Config, DataConfig


def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
    x = BatchNormalization(axis=Config.CHANNEL_AXIS)(x)
    x = Activation(Config.ACTIVATION)(x)
    return x


def inception_stem(input):
    x = conv_block(input, 8, 3, 3, subsample=(2, 2), border_mode='same')
    x = conv_block(x, 8, 3, 3, border_mode='same')
    x = conv_block(x, 16, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(x)
    x2 = conv_block(x, 24, 3, 3, subsample=(2, 2), border_mode='same')

    x = merge([x1, x2], mode='concat', concat_axis=Config.CHANNEL_AXIS)

    x1 = conv_block(x, 16, 1, 1)
    x1 = conv_block(x1, 24, 3, 3, border_mode='same')

    x2 = conv_block(x, 16, 1, 1)
    x2 = conv_block(x2, 16, 1, 3)
    x2 = conv_block(x2, 16, 3, 1)
    x2 = conv_block(x2, 24, 3, 3, border_mode='same')

    x = merge([x1, x2], mode='concat', concat_axis=Config.CHANNEL_AXIS)

    x1 = conv_block(x, 64, 3, 3, subsample=(2, 2), border_mode='same')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(x)

    x = merge([x1, x2], mode='concat', concat_axis=Config.CHANNEL_AXIS)
    return x


def inception_A(input):
    a1 = conv_block(input, 24, 1, 1)

    a2 = conv_block(input, 16, 1, 1)
    a2 = conv_block(a2, 24, 3, 3)

    a3 = conv_block(input, 16, 1, 1)
    a3 = conv_block(a3, 24, 3, 3)
    a3 = conv_block(a3, 24, 3, 3)

    a4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    a4 = conv_block(a4, 24, 1, 1)

    m = merge([a1, a2, a3, a4], mode='concat', concat_axis=Config.CHANNEL_AXIS)
    return m


def inception_B(input):
    b1 = conv_block(input, 64, 1, 1)

    b2 = conv_block(input, 64, 1, 1)
    b2 = conv_block(b2, 128, 1, 3)
    b2 = conv_block(b2, 144, 3, 1)

    b3 = conv_block(input, 64, 1, 1)
    b3 = conv_block(b3, 64, 3, 1)
    b3 = conv_block(b3, 128, 1, 3)
    b3 = conv_block(b3, 128, 3, 1)
    b3 = conv_block(b3, 144, 1, 3)

    b4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    b4 = conv_block(b4, 64, 1, 1)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=Config.CHANNEL_AXIS)
    return m


def inception_C(input):
    c1 = conv_block(input, 64, 1, 1)

    c2 = conv_block(input, 96, 1, 1)
    c2_1 = conv_block(c2, 64, 1, 3)
    c2_2 = conv_block(c2, 64, 3, 1)
    c2 = merge([c2_1, c2_2], mode='concat', concat_axis=Config.CHANNEL_AXIS)

    c3 = conv_block(input, 128, 1, 1)
    c3 = conv_block(c3, 144, 3, 1)
    c3 = conv_block(c3, 176, 1, 3)
    c3_1 = conv_block(c3, 64, 1, 3)
    c3_2 = conv_block(c3, 64, 3, 1)
    c3 = merge([c3_1, c3_2], mode='concat', concat_axis=Config.CHANNEL_AXIS)

    c4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c4 = conv_block(c4, 64, 1, 1)

    m = merge([c1, c2, c3, c4], mode='concat', concat_axis=Config.CHANNEL_AXIS)
    return m


def reduction_A(input):
    r1 = conv_block(input, 128, 3, 3, subsample=(2, 2), border_mode='same')

    r2 = conv_block(input, 64, 1, 1)
    r2 = conv_block(r2, 96, 3, 3)
    r2 = conv_block(r2, 128, 3, 3, subsample=(2, 2), border_mode='same')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(input)

    m = merge([r1, r2, r3], mode='concat', concat_axis=Config.CHANNEL_AXIS)
    return m


def reduction_B(input):
    r1 = conv_block(input, 64, 1, 1)
    r1 = conv_block(r1, 64, 3, 3, subsample=(2, 2), border_mode='same')

    r2 = conv_block(input, 96, 1, 1)
    r2 = conv_block(r2, 128, 1, 3)
    r2 = conv_block(r2, 144, 3, 1)
    r2 = conv_block(r2, 144, 3, 3, subsample=(2, 2), border_mode='same')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(input)

    m = merge([r1, r2, r3], mode='concat', concat_axis=Config.CHANNEL_AXIS)
    return m


def create_model():
    '''
    Creates a inception v4 network
    :param nb_classes: number of classes.txt
    :return: Keras Model with 1 input and 1 output
    '''

    init = Input(Config.INPUT_SHAPE)
    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Dropout
    x = Dropout(0.2)(x)

    # Output
    out = Dense(output_dim=DataConfig.get_number_of_classes(), activation='softmax')(x)

    model = Model(init, out, name='Inception-v4')

    return model


if __name__ == "__main__":
    inception_v4 = create_model()
    inception_v4.summary()

