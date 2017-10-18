import os
import string
from api.src.keras_extensions.metrics import f1


class Config(object):
    import keras.backend as K
    IMAGE_SIZE = 64
    EPSILON = 1e-6
    LOGGING = False
    ACTIVATION = 'elu'
    WEIGHT_INIT = 'he_normal'

    BACKEND = K.backend()
    # Keras specific
    if K.image_dim_ordering() == "th":
        INPUT_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)
        CHANNEL_AXIS = 1
    else:
        INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
        CHANNEL_AXIS = 3

    @staticmethod
    def get_config():
        return {
            'image_size': Config.IMAGE_SIZE,
            'epsilon': Config.EPSILON,
            'logging': Config.LOGGING,
            'activation': Config.ACTIVATION,
            'weight_init': Config.WEIGHT_INIT,
            'input_shape': Config.INPUT_SHAPE,
            'channel_axis': Config.CHANNEL_AXIS
        }


class TrainingConfig(object):
    from keras.optimizers import SGD, Adam

    NB_EPOCHS = 100
    NUM_WORKERS = 4
    BATCH_SIZE = 32
    TRAINING_PHASE = 1
    TESTING_PHASE = 0
    INITIAL_LEARNING_RATE = 0.03

    lr_schedule = [14, 20, 28]  # epoch_step

    PATHS = {
        'MODELS': os.path.join('api', 'models')
    }

    @staticmethod
    def schedule(epoch_idx):
        if (epoch_idx + 1) < TrainingConfig.lr_schedule[0]:
            return TrainingConfig.INITIAL_LEARNING_RATE
        elif (epoch_idx + 1) < TrainingConfig.lr_schedule[1]:
            return 0.005  # lr_decay_ratio = 0.2
        elif (epoch_idx + 1) < TrainingConfig.lr_schedule[2]:
            return 0.0005
        return 0.0001

    optimizer = 'sgd'
    available_optimizers = {
        'sgd': SGD(lr=INITIAL_LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True),
        'adam': Adam()
    }

    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy', f1]
    callbacks_monitor = 'loss'

    @staticmethod
    def get_config():
        return {
            'nb_epochs': TrainingConfig.NB_EPOCHS,
            'num_workers': TrainingConfig.NUM_WORKERS,
            'batch_size': TrainingConfig.BATCH_SIZE,
            'training_phase': TrainingConfig.TRAINING_PHASE,
            'testing_phase': TrainingConfig.TESTING_PHASE,
            'initial_lr': TrainingConfig.INITIAL_LEARNING_RATE,
            'lr_schedule': TrainingConfig.lr_schedule,
            'paths': TrainingConfig.PATHS,
            'optimizer': TrainingConfig.optimizer,
            'loss': TrainingConfig.loss,
            'callbacks_monitor': TrainingConfig.callbacks_monitor
        }


class DataConfig(object):
    from sklearn.preprocessing import LabelEncoder
    PATHS = {
        'RAW_DATA': os.path.join('data', 'raw'),
        'PROCESSED_DATA': os.path.join('data', 'processed'),
        'TRAINING_PROCESSED_DATA': os.path.join('data', 'processed', 'train'),
        'VALID_PROCESSED_DATA': os.path.join('data', 'processed', 'valid'),
        'LOG_DATA_IMAGES': os.path.join('data', 'logger', 'images'),
        'LOG_DATA_TEXT': os.path.join('data', 'logger', 'text')
    }

    # AVAILABLE_CHARS = string.digits + string.ascii_lowercase
    AVAILABLE_CHARS = string.ascii_lowercase
    CLASS_ENCODER = LabelEncoder()
    CLASS_ENCODER.fit(list(AVAILABLE_CHARS))

    @staticmethod
    def get_class(sign):
        return DataConfig.CLASS_ENCODER.transform([sign])

    @staticmethod
    def get_classes(signs):
        return DataConfig.CLASS_ENCODER.transform(signs)

    @staticmethod
    def get_one_hot(sign):
        x = [0.0] * len(DataConfig.AVAILABLE_CHARS)
        x[DataConfig.get_class(sign)[0]] = 1.0
        return x

    @staticmethod
    def get_number_of_classes():
        return len(DataConfig.CLASS_ENCODER.classes_)

    @staticmethod
    def get_config():
        return {
            'paths': DataConfig.PATHS,
            'available_chars': DataConfig.AVAILABLE_CHARS,
            'classes': DataConfig.CLASS_ENCODER.classes_,
        }


if __name__ == '__main__':
    AVAILABLE_CHARS = string.digits + string.ascii_lowercase
    print(AVAILABLE_CHARS)
    for c in AVAILABLE_CHARS:
        print(DataConfig.get_one_hot(c))
