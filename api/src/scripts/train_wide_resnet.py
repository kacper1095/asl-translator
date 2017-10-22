import os
import argparse
import datetime
import yaml
import api.src.common.initial_environment_config

from ..models.wrn_by_titu import create_wide_residual_network
from ..data_processing.data_generator import DataGenerator
from ..common.config import TrainingConfig, DataConfig, Config
from ..common.utils import print_info, ensure_dir

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, EarlyStopping

RUNNING_TIME = datetime.datetime.now().strftime("%H_%M_%d_%m_%y")


def train(num_epochs, batch_size, input_size, num_workers):
    if not Config.NO_SAVE:
        ensure_dir(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME))
    # model = create_model(get_spatial_transformer())
    model = create_wide_residual_network(Config.INPUT_SHAPE, N=2, k=4, dropout=0.4)
    model.summary()

    callbacks = [
        ModelCheckpoint(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'weights.h5'), save_best_only=True, monitor=TrainingConfig.callbacks_monitor),
        CSVLogger(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'history.csv')),
        # TensorBoard(log_dir=os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'tensorboard')),
        # LearningRateScheduler(TrainingConfig.schedule),
        EarlyStopping(patience=12)
    ] if not Config.NO_SAVE else []

    if not Config.NO_SAVE:
        introduced_change = input("What new was introduced?: ")
        with open(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'change.txt'), 'w') as f:
            f.write(introduced_change)

        with open(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'config.yml'), 'w') as f:
            yaml.dump(list([TrainingConfig.get_config(), Config.get_config(), DataConfig.get_config()]), f, default_flow_style=False)

    optimizer = TrainingConfig.optimizer
    data_generator_train = DataGenerator(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], batch_size, input_size, False)
    data_generator_valid = DataGenerator(DataConfig.PATHS['VALID_PROCESSED_DATA'], batch_size, input_size, True)
    model.compile(TrainingConfig.available_optimizers[optimizer], TrainingConfig.loss, metrics=TrainingConfig.metrics)

    model.fit_generator(data_generator_train, samples_per_epoch=data_generator_train.samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=data_generator_valid, nb_val_samples=data_generator_valid.samples_per_epoch,
                        callbacks=callbacks)


def main(args):
    print_info("Training")
    train(args.num_epochs, args.batch_size, args.input_size, args.num_workers)
    print_info("Finished")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Script performing training')
    argparser.add_argument('--num_epochs', default=TrainingConfig.NB_EPOCHS, type=int, help='Number of training epochs')
    argparser.add_argument('--num_workers', type=int, default=TrainingConfig.NUM_WORKERS, help='Number of workers during training')
    argparser.add_argument('--batch_size', type=int, default=TrainingConfig.BATCH_SIZE, help='Batch size')
    argparser.add_argument('--input_size', type=int, default=Config.IMAGE_SIZE, help='Image size to input')
    arguments = argparser.parse_args()
    main(arguments)
