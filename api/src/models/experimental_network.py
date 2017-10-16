from keras.layers import merge, Input, Convolution2D, BatchNormalization, Activation, AveragePooling2D, Dense, Flatten
from keras.models import Model
from api.src.common.config import Config, DataConfig


def bn_block(x, filters, kernel_size, subsample=(1, 1)):
    x = BatchNormalization(Config.CHANNEL_AXIS)(x)
    x = Activation(Config.ACTIVATION)(x)
    x = Convolution2D(filters, kernel_size, kernel_size, init=Config.WEIGHT_INIT, border_mode='same', subsample=subsample)(x)
    return x


def max_block(inputs, number_of_parallel_convos):
    parallels = [bn_block(inputs[-1], 3, 3) for _ in range(number_of_parallel_convos)]
    # parallels = [bn_block(x, 3, 3) for x in parallels]
    max_out = merge(parallels, mode='max')
    concat_out = merge(inputs + [max_out], mode='concat', concat_axis=Config.CHANNEL_AXIS)
    return concat_out


def create_model():
    inputs = Input(Config.INPUT_SHAPE)
    convo = bn_block(inputs, 3, 5, (2, 2))

    nb_of_parallel = [32, 32, 48, 64]
    nb_repetitions = [2, 2, 2, 3]
    convos_in_level = [convo]
    for p, r in zip(nb_of_parallel, nb_repetitions):
        for _ in range(r):
            convo = max_block(convos_in_level, p)
            convos_in_level.append(convo)
        convo = AveragePooling2D()(convos_in_level[-1])
        convos_in_level = [convo]
    x = Flatten()(convo)
    x = Dense(DataConfig.get_number_of_classes(), activation='softmax')(x)

    model = Model(inputs, x, 'experimental')
    return model


if __name__ == '__main__':
    import numpy as np
    # create_model().summary()
    model = create_model()
    model.compile('sgd', 'categorical_crossentropy', metrics=['acc'])
    print('training')
    model.fit(np.random.random((32, 3, 64, 64)), np.round(np.random.random((32, 26))))
