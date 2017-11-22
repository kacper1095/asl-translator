import os
import datetime
import yaml
import api.src.common.initial_environment_config

from ..models.wrn_by_titu import create_wide_residual_network
from ..data_processing.data_generator import DataGenerator
from ..common.config import TrainingConfig, DataConfig, Config
from ..common.utils import print_info, ensure_dir
from .plot_trainings import get_description_string

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

RUNNING_TIME = datetime.datetime.now().strftime("%H_%M_%d_%m_%y")


def data():
    data_generator_train = DataGenerator(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], TrainingConfig.BATCH_SIZE, Config.IMAGE_SIZE, False,
                                         without_preprocessing=True)
    data_generator_valid = DataGenerator(DataConfig.PATHS['VALID_PROCESSED_DATA'], TrainingConfig.BATCH_SIZE, Config.IMAGE_SIZE, True,
                                         without_preprocessing=True)
    return data_generator_train, data_generator_valid


def train(data_generator_train, data_generator_valid):
    if not Config.NO_SAVE:
        ensure_dir(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME))

    if conditional({{choice([False, True])}}):  # using pretrained weights
        model = create_wide_residual_network(Config.INPUT_SHAPE, N=2, k=8, dropout={{uniform(0.2, 0.8)}},
                                             path_weights=os.path.join(DataConfig.PATHS['PRETRAINED_MODEL_FOLDER'],
                                                                       'WRN-16-8 Weights.h5'),
                                             layer_to_stop_freezing={{choice([
                                                 'convolution2d_1', 'merge_2', 'convolution2d_3', 'merge_1', ''
                                             ])}})
    else:
        model = create_wide_residual_network(Config.INPUT_SHAPE, N={{choice([1, 2, 3])}},
                                             k={{choice([1, 2, 4, 8])}},
                                             dropout={{uniform(0.2, 0.8)}})

    callbacks = [
        ModelCheckpoint(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'weights.h5'), save_best_only=True,
                        monitor=TrainingConfig.callbacks_monitor),
        CSVLogger(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'history.csv')),
        EarlyStopping(patience=12)
    ] if not Config.NO_SAVE else []

    if not Config.NO_SAVE:
        with open(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'config.yml'), 'w') as f:
            yaml.dump(list([TrainingConfig.get_config(), Config.get_config(), DataConfig.get_config()]), f,
                      default_flow_style=False)

        with open(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'model.txt'), 'w') as f:
            f.write(get_description_string(model))

    optimizer = TrainingConfig.optimizer
    model.compile(TrainingConfig.available_optimizers[optimizer], TrainingConfig.loss, metrics=TrainingConfig.metrics)

    model.fit_generator(data_generator_train, samples_per_epoch=data_generator_train.samples_per_epoch,
                        nb_epoch=TrainingConfig.NB_EPOCHS,
                        validation_data=data_generator_valid, nb_val_samples=data_generator_valid.samples_per_epoch,
                        callbacks=callbacks, class_weight=data_generator_train.get_class_weights)
    loss, acc, _, _ = model.evaluate_generator(data_generator_valid, data_generator_valid.samples_per_epoch)
    print('Test accuracy: ', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def main():
    print_info("Training")
    best_run, best_model = optim.minimize(model=train,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=Trials())
    print(best_run)
    best_model.save_weights(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'best_of_all.h5'))
    print_info("Finished")


if __name__ == '__main__':
    main()
