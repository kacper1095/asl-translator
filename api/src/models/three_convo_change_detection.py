from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Lambda, Activation, Input, Dropout, Flatten, \
    Dense
from api.src.common.config import Config, DataConfig

import keras.backend as K


def bn_convo(x, filters, kernel_size, use_dropout=False):
    x = Convolution2D(filters, kernel_size, kernel_size,
                      border_mode='same', init=Config.WEIGHT_INIT)(x)
    x = BatchNormalization(axis=Config.CHANNEL_AXIS)(x)
    x = Activation(Config.ACTIVATION)(x)
    if use_dropout:
        x = Dropout(0.0)(x)
    return x


def base_model():
    inputs = Input(Config.INPUT_SHAPE)
    filters = [32, 64, 128, 128]
    repetitions = [2, 2, 3, 3]

    x = Convolution2D(16, 5, 5, init=Config.WEIGHT_INIT,
                      border_mode='same', subsample=(2, 2))(inputs)

    for index, (f, r) in enumerate(zip(filters, repetitions)):
        for _ in range(r):
            x = bn_convo(x, f, 3, use_dropout=True)
        if index == len(filters) - 1:
            break
        x = MaxPooling2D((3, 3), (2, 2))(x)

    x = Flatten()(x)
    model = Model(inputs, x, '3_convo')
    return model


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def create_model():
    inputs_1 = Input(Config.INPUT_SHAPE)
    inputs_2 = Input(Config.INPUT_SHAPE)
    base = base_model()
    x_1 = base(inputs_1)
    x_2 = base(inputs_2)

    x = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([x_1, x_2])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2, activation='softmax')(x)
    return Model([inputs_1, inputs_2], x)


if __name__ == '__main__':
    create_model().summary()
