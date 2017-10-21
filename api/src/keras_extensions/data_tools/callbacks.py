import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime

from keras.callbacks import Callback
import keras.callbacks as callbacks


class PlotCallback(Callback):
    """

    """

    def __init__(self, folder_path, plot_name):
        super(PlotCallback, self).__init__()

        self.plotter = plt
        self.folder_path = folder_path
        self.plot_name = plot_name
        self.iteration_number = 0
        self.metrics = {}
        self.iterations_numbers = {}

    def on_train_begin(self, logs=None):
        self.metrics = {}
        self.iterations_numbers = {}

    def on_batch_end(self, batch, logs=None):
        self.metrics.setdefault('train_loss', []).append(logs.setdefault('loss', 0))
        self.iterations_numbers.setdefault('train_loss', []).append(self.iteration_number)
        self.iteration_number += 1

    def on_epoch_end(self, epoch, logs=None):
        self.metrics.setdefault('val_loss', []).append(logs.setdefault('val_loss', 0))
        self.iterations_numbers.setdefault('val_loss', []).append(self.iteration_number)
        self.save_plot()

    def save_plot(self):
        self.plotter.plot(self.iterations_numbers['train_loss'], self.metrics['train_loss'])
        self.plotter.plot(self.iterations_numbers.setdefault('val_loss', []), self.metrics.setdefault('val_loss', []))
        self.plotter.title('Model loss')
        self.plotter.ylabel('Loss')
        self.plotter.xlabel('Iteration')
        self.plotter.legend(self.metrics.keys(), loc='upper right')
        self.plotter.savefig(
            os.path.join(self.folder_path, 'history_{}.png'.format(self.plot_name)))


class SnapshotModelCheckpoint(Callback):
    """Callback that saves the snapshot weights of the model.
    Saves the model weights on certain epochs (which can be considered the
    snapshot of the model at that epoch).
    Should be used with the cosine annealing learning rate schedule to save
    the weight just before learning rate is sharply increased.
    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            # print("Saved snapshot at weights/%s_%d.h5" % (self.fn_prefix, epoch))


class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.
    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """

    def __init__(self, nb_epochs, nb_snapshots, checkpoint_path, init_lr=0.1):
        """
        Initialize a snapshot callback builder.
        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.checkpoint_path = checkpoint_path

    def get_callbacks(self, model_prefix='Model'):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.
        Args:
            model_prefix: prefix for the filename of the weights.
        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """

        callback_list = [callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path, "%s-Best.h5") % model_prefix,
                                                   monitor="val_loss",
                                                   save_best_only=True, save_weights_only=False),
                         callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
                         # callbacks.CSVLogger(os.path.join(self.checkpoint_path, 'history.csv')),
                         SnapshotModelCheckpoint(self.T, self.M,
                                                 fn_prefix=os.path.join(self.checkpoint_path, '%s') % model_prefix)]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)
