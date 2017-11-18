from keras.models import Model
from keras.layers import BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, Convolution2D, Activation, merge, Input, Dropout

from api.src.common.config import Config, DataConfig


def bn_convo(x, kernel_size, nb_filters, border_mode='same', subsample=(1, 1)):
    x = Convolution2D(nb_filters, kernel_size, kernel_size, subsample=subsample, init=Config.WEIGHT_INIT, border_mode=border_mode)(x)
    x = BatchNormalization(axis=Config.CHANNEL_AXIS)(x)
    x = Activation('elu')(x)
    return x


def competetive_block(x, nb_filters, nb_convos):
    layers = [bn_convo(x, 2 * i + 1, nb_filters) for i in range(nb_convos)]
    return merge(layers, mode='max')


def big_block(x, nb_of_convos, nb_filters, use_dropout_and_pooling=True):
    for nb_convos, nb_f in zip(nb_of_convos, nb_filters):
        x = competetive_block(x, nb_f, nb_convos)
    if use_dropout_and_pooling:
        x = MaxPooling2D((3, 3), (2, 2))(x)
        x = Dropout(0.5)(x)
    return x


def create_model():
    inp = Input(Config.INPUT_SHAPE)
    x = bn_convo(inp, 7, 16, border_mode='valid', subsample=(2, 2))
    x = big_block(x, [4, 2, 2], [192, 160, 96])
    x = big_block(x, [4, 2, 2], [192, 192, 192])
    x = big_block(x, [3, 2, 2], [192, 192, DataConfig.get_number_of_classes()], use_dropout_and_pooling=False)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    model = Model(inp, x, name='competetive')
    return model


if __name__ == '__main__':
    create_model().summary()

