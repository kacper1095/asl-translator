from ..common import initial_environment_config

from ..common.config import DataConfig, Config

from keras.models import Model
from keras.layers import Dense, Input, Lambda, BatchNormalization, Flatten
from keras.regularizers import l2

import keras.backend as K


def create_model(input_shape):
    inputs = Input(input_shape)
    x = Dense(DataConfig.get_number_of_classes(), activation='softmax')(inputs)
    model = Model(input=inputs, output=x)
    return model
