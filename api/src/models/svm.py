from ..common import initial_environment_config

from ..common.config import DataConfig, Config

from keras.models import Model
from keras.layers import Dense, Input, Lambda, BatchNormalization
from keras.regularizers import l2

import keras.backend as K


def gaussian_radial_basis_function(x):
    means = K.mean(x, axis=0, keepdims=True)
    return K.exp(-(K.abs(means - x)) ** 2)


def create_model(input_vector_length):
    inputs = Input(shape=(input_vector_length,))
    x = BatchNormalization()(inputs)
    x = Dense(input_vector_length * 2, activation='sigmoid')(x)
    x = Dense(DataConfig.get_number_of_classes(), W_regularizer=l2(0.01), activation='softmax')(x)
    model = Model(input=inputs, output=x)
    return model
