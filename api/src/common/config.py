import keras.backend as K

from keras.optimizers import SGD


class Config(object):

    IMAGE_SIZE = 32
    NB_CLASSES = 24

    # Keras specific
    if K.image_dim_ordering() == "th":
        INPUT_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)
        CHANNEL_AXIS = 1
    else:
        INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
        CHANNEL_AXIS = 3


class TrainingConfig(object):
    nb_epochs = 200
    lr_schedule = [60, 120, 160]  # epoch_step

    @staticmethod
    def schedule(epoch_idx):
        if (epoch_idx + 1) < TrainingConfig.lr_schedule[0]:
            return 0.1
        elif (epoch_idx + 1) < TrainingConfig.lr_schedule[1]:
            return 0.02  # lr_decay_ratio = 0.2
        elif (epoch_idx + 1) < TrainingConfig.lr_schedule[2]:
            return 0.004
        return 0.0008

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)


class DataConfig(object):
    pass
