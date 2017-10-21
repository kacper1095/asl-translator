from api.src.common import initial_environment_config
from api.src.common.config import Config, DataConfig

from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16


def create_model():
    init = Input(Config.INPUT_SHAPE)
    model = VGG16(include_top=False, input_tensor=init)
    for layer in model.layers:
        if layer.name == 'block3_pool':
            break
        layer.trainable = False
    x = GlobalAveragePooling2D()(model.output)
    x = Dropout(0.2)(x)
    x = Dense(DataConfig.get_number_of_classes(), activation='softmax')(x)

    model = Model(init, x, 'vgg16')
    return model


if __name__ == '__main__':
    create_model().summary()

