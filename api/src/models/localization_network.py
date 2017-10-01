from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model
from api.src.keras_extensions.layers import SpatialTransformer
from api.src.common.config import Config


def bn_convo(filters, kernel_size, stride):
    def f(_input):
        x = BatchNormalization(axis=Config.CHANNEL_AXIS)(_input)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=stride, padding='same',
                   kernel_initializer='he_normal')(x)
        return x
    return f


def create_localization_net():
    input_tensor = Input(Config.INPUT_SHAPE)
    x = bn_convo(32, 7, (2, 2))(input_tensor)
    x = bn_convo(64, 3, (1, 1))(x)
    x = MaxPooling2D((2, 2))(x)
    x = bn_convo(64, 3, (1, 1))(x)
    x = MaxPooling2D((2, 2))(x)
    x = bn_convo(128, 3, (1, 1))(x)
    x = MaxPooling2D((2, 2))(x)
    x = bn_convo(128, 3, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(6, kernel_initializer='he_normal')(x)
    model = Model(input_tensor, x, 'localization_net')

    return model


def get_spatial_transformer():
    return SpatialTransformer(create_localization_net())

if __name__ == '__main__':
    create_localization_net().summary()
