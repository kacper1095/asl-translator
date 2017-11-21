from ..common import initial_environment_config

from ..common.config import DataConfig, Config

from keras.models import Model
from keras.layers import Dense, Input, Lambda, BatchNormalization, Flatten
from keras.regularizers import l2

import keras.backend as K


def gaussian_radial_basis_function(x):
    means = K.mean(x, axis=0, keepdims=True)
    return K.exp(-(K.abs(means - x)) ** 2)


def create_model(input_shape=Config.INPUT_SHAPE):
    inputs = Input(input_shape)
    x = BatchNormalization(axis=-1)(inputs)
    x = Dense(1024, activation='elu')(x)
    x = Dense(DataConfig.get_number_of_classes(), activation='softmax')(x)
    model = Model(input=inputs, output=x)
    return model
