from ..common import initial_environment_config

from ..common.config import DataConfig, Config

from keras.models import Model
from keras.layers import Dense, Input, Lambda, BatchNormalization, Flatten
from keras.regularizers import l2

import keras.backend as K


def gaussian_radial_basis_function(x):
    means = K.mean(x, axis=0, keepdims=True)
    return K.exp(-(K.abs(means - x)) ** 2)


def create_model():
    inputs = Input(Config.INPUT_SHAPE)
    x = Flatten()(inputs)
    x = Dense(sum(s for s in Config.INPUT_SHAPE[1:]) // 2, activation='sigmoid')(x)
    x = Dense(DataConfig.get_number_of_classes(), activation='softmax')(x)
    model = Model(input=inputs, output=x)
    return model
