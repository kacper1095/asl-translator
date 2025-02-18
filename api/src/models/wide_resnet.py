"""
Code from my Wide Residual Network repository : https://github.com/titu1994/Wide-Residual-Networks
"""
from keras.models import Model
from keras.layers import Input, merge, Activation, Dropout, Flatten, Dense, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from api.src.common.config import Config, DataConfig


def initial_conv(input):
    channel_axis = 1 if K.image_dim_ordering() == "th" else -1
    x = Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(Config.ACTIVATION)(x)

    return x


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 16 * k,
    # else create convolution2d for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 16 * k:
            init = Convolution2D(16 * k, 1, 1,
                                 activation='linear',
                                 border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 16 * k:
            init = Convolution2D(16 * k, 1, 1,
                                 activation='linear',
                                 border_mode='same')(init)

    x = Convolution2D(16 * k, 3, 3,
                      border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(Config.ACTIVATION)(x)

    if dropout > 0.0: x = SpatialDropout2D(dropout)(x)
    print(dropout)

    x = Convolution2D(16 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(Config.ACTIVATION)(x)

    m = merge([init, x], mode='sum')
    return m

def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 32 * k, else create convolution2d for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 32 * k:
            init = Convolution2D(32 * k, 1, 1, activation='linear', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 32 * k:
            init = Convolution2D(32 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Convolution2D(32 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(Config.ACTIVATION)(x)

    if dropout > 0.0: x = SpatialDropout2D(dropout)(x)
    print(dropout)

    x = Convolution2D(32 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(Config.ACTIVATION)(x)

    m = merge([init, x], mode='sum')
    return m

def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 64 * k, else create convolution2d for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 64 * k:
            init = Convolution2D(64 * k, 1, 1, activation='linear', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 64 * k:
            init = Convolution2D(64 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Convolution2D(64 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(Config.ACTIVATION)(x)

    if dropout > 0.0: x = SpatialDropout2D(dropout)(x)
    print(dropout)

    x = Convolution2D(64 * k, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(Config.ACTIVATION)(x)

    m = merge([init, x], mode='sum')
    return m


def create_wide_residual_network(input_dim, N=2, k=1, dropout=0.0, verbose=1, path_weights=None, layer_to_stop_freezing='merge_4'):
    """
    Creates a Wide Residual Network with specified parameters
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2r
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    for i in range(N):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = MaxPooling2D((2,2))(x)

    for i in range(N):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = MaxPooling2D((2,2))(x)

    for i in range(N):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(DataConfig.get_number_of_classes(), activation='softmax', name='classifier')(x)

    model = Model(ip, x)
    if path_weights is not None:
        model.load_weights(path_weights, by_name=True)
        for layer in model.layers:
            if layer.name == layer_to_stop_freezing:
                break
            layer.trainable = False

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model

    init = (3, 64, 64)

    model = create_wide_residual_network(init, N=2, k=8, dropout=0.25, path_weights=DataConfig.PATHS['PRETRAINED_MODEL_FOLDER'] + '/WRN-16-8 Weights.h5',
                                         layer_to_stop_freezing='merge_2')


    model.summary()
    # with open('architecture.json', 'w') as f:
    #     f.write(model.to_json())
    # plot(model, "WRN-28-10.png", show_shapes=True, show_layer_names=True)