from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2

from api.src.models.localization_network import create_localization_net, get_spatial_transformer
from api.src.common.config import Config, DataConfig

import logging

logging.basicConfig(level=logging.DEBUG)

depth = 16  # table 5 on page 8 indicates best value (4.17) CIFAR-10
k = 8  # 'widen_factor'; table 5 on page 8 indicates best value (4.17) CIFAR-10
dropout_probability = 0  # table 6 on page 10 indicates best value (4.17) CIFAR-10

weight_decay = 0.0005  # page 10: "Used in all experiments"

nb_epochs = 200
lr_schedule = [60, 120, 160]  # epoch_step


def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008


sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

use_bias = False  # following functions 'FCkernel_initializer(model)' and 'DisableBias(model)' in utils.lua
weight_kernel_initializer = "he_normal"  # follows the 'MSRkernel_initializer(model)' function in utils.lua


# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               strides="(stride_vertical,stride_horizontal)",
        #               padding="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [[3, 3, stride, "same"],
                       [3, 3, (1, 1), "same"]]

        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=Config.CHANNEL_AXIS)(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=Config.CHANNEL_AXIS)(net)
                    convs = Activation("relu")(convs)
                convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                               strides=v[2],
                               padding=v[3],
                               kernel_initializer=weight_kernel_initializer,
                               kernel_regularizer=l2(weight_decay),
                               use_bias=use_bias)(convs)
            else:
                convs = BatchNormalization(axis=Config.CHANNEL_AXIS)(convs)
                convs = Activation("relu")(convs)
                if dropout_probability > 0:
                    convs = Dropout(dropout_probability)(convs)
                convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                               strides=v[2],
                               padding=v[3],
                               kernel_initializer=weight_kernel_initializer,
                               kernel_regularizer=l2(weight_decay),
                               use_bias=use_bias)(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),
                              strides=stride,
                              padding="same",
                              kernel_initializer=weight_kernel_initializer,
                              kernel_regularizer=l2(weight_decay),
                              use_bias=use_bias)(net)
        else:
            shortcut = net

        return Add()([convs, shortcut])

    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2, int(count + 1)):
            net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
        return net

    return f


def create_model(spatial_network=None):
    logging.debug("Creating model...")

    assert ((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    inputs = Input(shape=Config.INPUT_SHAPE)

    n_stages = [16, 16 * k, 32 * k, 64 * k]
    if spatial_network is not None:
        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),
                       strides=(1, 1),
                       padding="same",
                       kernel_initializer=weight_kernel_initializer,
                       kernel_regularizer=l2(weight_decay),
                       use_bias=use_bias)(spatial_network(inputs))  # "One conv at the beginning (spatial size: 32x32)"
    else:
        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),
                       strides=(1, 1),
                       padding="same",
                       kernel_initializer=weight_kernel_initializer,
                       kernel_regularizer=l2(weight_decay),
                       use_bias=use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"


    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(
        conv1)  # "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(
        conv2)  # "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(
        conv3)  # "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=Config.CHANNEL_AXIS)(conv4)
    relu = Activation("relu")(batch_norm)

    # Classifier block
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
    flatten = Flatten()(pool)
    predictions = Dense(units=DataConfig.get_number_of_classes(), kernel_initializer=weight_kernel_initializer, use_bias=use_bias,
                        kernel_regularizer=l2(weight_decay), activation="softmax")(flatten)

    model = Model(inputs=inputs, outputs=predictions)
    return model


if __name__ == '__main__':
    create_model(get_spatial_transformer()).summary()
