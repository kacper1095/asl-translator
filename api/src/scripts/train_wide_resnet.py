import os
import argparse
import datetime
import api.src.common.initial_environment_config

from ..models.wide_resnet import create_model, get_spatial_transformer
from ..data_processing.data_generator import DataGenerator
from ..common.config import TrainingConfig, DataConfig, Config
from ..common.utils import print_info, ensure_dir

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler

RUNNING_TIME = datetime.datetime.now().strftime("%H_%M_%d_%m_%y")


def train(num_epochs, batch_size, input_size, num_workers):
    ensure_dir(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME))
    # model = create_model(get_spatial_transformer())
    model = create_model()

    callbacks = [
        ModelCheckpoint(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'weights.h5'), save_best_only=True, monitor=TrainingConfig.callbacks_monitor),
        CSVLogger(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'history.csv')),
        TensorBoard(log_dir=os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'tensorboard')),
        LearningRateScheduler(TrainingConfig.schedule)
    ]

    optimizer = TrainingConfig.optimizer
    data_generator_train = DataGenerator(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], batch_size, input_size)
    data_generator_valid = DataGenerator(DataConfig.PATHS['VALID_PROCESSED_DATA'], batch_size, input_size)
    model.compile(optimizer, TrainingConfig.loss, metrics=TrainingConfig.metrics)

    model.fit_generator(data_generator_train, steps_per_epoch=data_generator_train.number_of_steps, epochs=num_epochs,
                        callbacks=callbacks, workers=num_workers, max_queue_size=24,
                        validation_data=data_generator_valid, validation_steps=data_generator_valid.number_of_steps)


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
