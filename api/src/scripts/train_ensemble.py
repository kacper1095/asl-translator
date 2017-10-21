import argparse
import os
import api.src.common.initial_environment_config
import datetime

import yaml
from ..data_processing.data_generator import DataGenerator
from ..common.config import TrainingConfig, DataConfig, Config
from ..common.utils import print_info, ensure_dir

from api.src.keras_extensions.data_tools.callbacks import SnapshotCallbackBuilder
from api.src.models import wrn_by_titu as WRN

RUNNING_TIME = datetime.datetime.now().strftime("%H_%M_%d_%m_%y")


def train(num_epochs, batch_size, input_size, M, alpha_zero, wrn_N, wrn_k, num_workers):
    if not Config.NO_SAVE:
        ensure_dir(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME))
    snapshot = SnapshotCallbackBuilder(num_epochs, M, os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME) , alpha_zero)
    model = WRN.create_wide_residual_network(Config.INPUT_SHAPE, N=wrn_N, k=wrn_k, dropout=0.00)
    model_prefix = 'wrn-%d-%d' % (wrn_N * 6 + 4, wrn_k)

    if not Config.NO_SAVE:
        introduced_change = input("What new was introduced?: ")
        with open(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'change.txt'), 'w') as f:
            f.write(introduced_change)

        with open(os.path.join(TrainingConfig.PATHS['MODELS'], RUNNING_TIME, 'config.yml'), 'w') as f:
            yaml.dump(list([TrainingConfig.get_config(), Config.get_config(), DataConfig.get_config()]), f, default_flow_style=False)

    optimizer = TrainingConfig.optimizer
    data_generator_train = DataGenerator(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], batch_size, input_size, False, False)
    data_generator_valid = DataGenerator(DataConfig.PATHS['VALID_PROCESSED_DATA'], batch_size, input_size, True, False)
    model.compile(optimizer, TrainingConfig.loss, metrics=TrainingConfig.metrics)

    model.fit_generator(data_generator_train, samples_per_epoch=data_generator_train.samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=data_generator_valid, nb_val_samples=data_generator_valid.samples_per_epoch,
                        callbacks=snapshot.get_callbacks(model_prefix=model_prefix), nb_worker=num_workers)


def main(args):
    print_info("Training")
    train(args.num_epochs, args.batch_size, args.input_size, args.M, args.alpha_zero, args.wrn_N, args.wrn_k, args.num_workers)
    print_info("Finished")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Script performing training')
    argparser.add_argument('--num_epochs', default=TrainingConfig.NB_EPOCHS, type=int,
                           help='Number of training epochs')
    argparser.add_argument('--M', type=int, default=5, help='Number of snapshots')
    argparser.add_argument('--num_workers', type=int, default=TrainingConfig.NUM_WORKERS,
                           help='Number of workers during training')
    argparser.add_argument('--batch_size', type=int, default=TrainingConfig.BATCH_SIZE, help='Batch size')
    argparser.add_argument('--input_size', type=int, default=Config.IMAGE_SIZE, help='Image size to input')

    argparser.add_argument('--alpha_zero', type=float, default=0.1, help='Initial learning rate')

    # Wide ResNet Parameters
    argparser.add_argument('--wrn_N', type=int, default=2, help='Number of WRN blocks. Computed as N = (n - 4) / 6.')
    argparser.add_argument('--wrn_k', type=int, default=4, help='Width factor of WRN')
    arguments = argparser.parse_args()
    main(arguments)