from ..common import initial_environment_config

from ..common.config import DataConfig, TrainingConfig

from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l2


def create_model(input_shape):
    inputs = Input(input_shape)
    x = Dense(DataConfig.get_number_of_classes(), activation='linear',
              W_regularizer=l2(TrainingConfig.SVM_WEIGHT_REGULARIZER))(inputs)
    model = Model(input=inputs, output=x)
    return model


