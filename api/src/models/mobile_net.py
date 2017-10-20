from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, Dense, GlobalAveragePooling2D, Activation, Input, Dropout, Flatten
from api.src.common.config import Config, DataConfig


def convo_block(x, filters, kernel_size, strides=(1, 1)):
    x = Convolution2D(filters, kernel_size, kernel_size, subsample=strides,
                      border_mode='same', init=Config.WEIGHT_INIT, bias=False)(x)
    x = BatchNormalization(axis=Config.CHANNEL_AXIS)(x)
    x = Activation(Config.ACTIVATION)(x)
    return x


def almost_depthwise_block(x, filters, strides=(1, 1)):
    x = Convolution2D(filters, 3, 3, init=Config.WEIGHT_INIT,
                      border_mode='same', subsample=strides, bias=False)(x)
    x = BatchNormalization(axis=Config.CHANNEL_AXIS)(x)
    x = Activation(Config.ACTIVATION)(x)

    x = Convolution2D(filters, 1, 1, init=Config.WEIGHT_INIT,
                      border_mode='same', bias=False)(x)
    x = BatchNormalization(axis=Config.CHANNEL_AXIS)(x)
    x = Activation(Config.ACTIVATION)(x)
    return x


def create_model():
    inputs = Input(Config.INPUT_SHAPE)

    x = convo_block(inputs, 8, 3, strides=(2, 2))
    x = almost_depthwise_block(x, 16)
    x = almost_depthwise_block(x, 32, (2, 2))
    x = almost_depthwise_block(x, 32)
    x = almost_depthwise_block(x, 48, (2, 2))
    x = almost_depthwise_block(x, 48)
    x = almost_depthwise_block(x, 64, (2, 2))
    x = almost_depthwise_block(x, 64)
    x = almost_depthwise_block(x, 64)
    x = almost_depthwise_block(x, 64)
    x = almost_depthwise_block(x, 64)
    x = almost_depthwise_block(x, 64)
    x = almost_depthwise_block(x, 128, (2, 2))
    x = almost_depthwise_block(x, 128)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.15)(x)
    x = Dense(DataConfig.get_number_of_classes(), activation='softmax')(x)

    model = Model(inputs, x, 'mobile_net')
    return model


if __name__ == '__main__':
    create_model().summary()
