import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import datetime

from keras.callbacks import Callback


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
