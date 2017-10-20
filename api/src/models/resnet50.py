from api.src.common import initial_environment_config
from api.src.common.config import Config, DataConfig

from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50


def create_model():
    init = Input(Config.INPUT_SHAPE)
    model = ResNet50(include_top=False, input_tensor=init)
    for layer in model.layers:
        if layer.name == 'res3a_branch2a':
            break
        layer.trainable = False
    x = GlobalAveragePooling2D()(model.output)
    x = Dropout(0.2)(x)
    x = Dense(DataConfig.get_number_of_classes(), activation='softmax')(x)

    model = Model(init, x, 'resnet50')
    return model


if __name__ == '__main__':
    create_model().summary()

