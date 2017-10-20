from ..common import initial_environment_config

from ..common.config import DataConfig, TrainingConfig, Config

from keras.models import Model
from keras.layers import Dense, Input, Convolution2D, BatchNormalization, Activation, merge, AveragePooling2D, Flatten, GlobalAveragePooling2D
from keras.regularizers import l2

import keras.backend as K


def convo_block(x, filters):
    x = Convolution2D(filters, 3, 3, init=Config.WEIGHT_INIT, border_mode='same')(x)
    x = Activation(Config.ACTIVATION)(x)
    return x


def residual_block(x, filters):
    inputs = x
    x = convo_block(x, filters)
    x = convo_block(x, filters)
    return merge([x, inputs], mode='sum')


def subsampling_block(x, filters):
    inputs = x
    x = Convolution2D(filters, 1, 1, init=Config.WEIGHT_INIT, subsample=(2, 2))(x)
    x = convo_block(x, filters)
    x = convo_block(x, filters)
    shortcut = Convolution2D(filters, 1, 1, init=Config.WEIGHT_INIT, subsample=(2, 2))(inputs)
    return merge([x, shortcut], mode='sum')


def create_model():
    inputs = Input(Config.INPUT_SHAPE)
    num_of_blocks = [1, 1, 1, 1]
    num_of_filters = [48, 64, 128, 96]
    x = inputs
    for filters, blocks in zip(num_of_filters, num_of_blocks):
        x = convo_block(x, filters)
        for _ in range(blocks):
            x = residual_block(x, filters)
        x = subsampling_block(x, filters)

    x = GlobalAveragePooling2D()(x)
    x = Dense(DataConfig.get_number_of_classes(), activation='linear',
              W_regularizer=l2(TrainingConfig.SVM_WEIGHT_REGULARIZER))(x)
    model = Model(input=inputs, output=x)
    return model


if __name__ == '__main__':
    create_model().summary()
