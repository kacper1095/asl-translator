from ..common import initial_environment_config
from ..common.config import TrainingConfig

import glob
import os
import csv
import numpy as np
import matplotlib
import sys
import tqdm
from io import StringIO
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from keras.models import load_model

REQUIRED_COLUMNS = ['categorical_accuracy', 'epoch', 'loss', 'val_categorical_accuracy', 'val_loss']
OPTIONAL_COLUMNS = ['f1', 'val_f1']


def plot():
    for history_file in tqdm.tqdm(list_training_folders()):
        with open(history_file) as f:
            plot_csv_file(csv.reader(f), history_file)
    for weight in tqdm.tqdm(list_model_folders()):
        try:
            save_model_summary(weight)
        except ValueError as e:
            print(e)


def save_model_summary(model_path):
    model = load_model(model_path, custom_objects={'f1': lambda x, y: x})
    orig_stdout = sys.stdout
    output_buf = StringIO()
    sys.stdout = output_buf

    model.summary()
    sys.stdout = orig_stdout
    description = output_buf.getvalue()

    path = os.path.dirname(model_path)
    filename = os.path.basename(model_path).split('.')[0] + '.txt'
    with open(os.path.join(path, filename), 'w') as f:
        f.write(description)

def list_training_folders():
    return glob.glob(os.path.join(TrainingConfig.PATHS['MODELS'], '**', '*.csv'))


def list_model_folders():
    return glob.glob(os.path.join(TrainingConfig.PATHS['MODELS'], '**', '*.h5'))


def plot_csv_file(reader, path):
    categories = []
    loss = []
    loss_val = []
    accuracy = []
    accuracy_val = []
    f1 = []
    f1_val = []
    nb_epochs = 0
    for i, row in enumerate(reader):

        if i == 0:
            if any([column_name not in row for column_name in REQUIRED_COLUMNS]):
                return
            categories.extend(row)
        else:
            nb_epochs += 1
            accuracy.append(float(row[1]))
            f1.append(float(row[2]))
            loss.append(float(row[3]))
            accuracy_val.append(float(row[4]))
            f1_val.append(float(row[5]))
            loss_val.append(float(row[6]))
    if len(loss) != 0 and len(loss_val) != 0:
        save_path = os.path.dirname(path)
        plot_data([loss, loss_val], 'Loss', os.path.join(save_path, 'loss.png'), 'loss')
        plot_data([accuracy, accuracy_val], 'Accuracy', os.path.join(save_path, 'accuracy.png'), 'accuracy')
        if len(f1) > 0:
            plot_data([f1, f1_val], 'F1 score', os.path.join(save_path, 'f1.png'), 'f1 score')
        with open(os.path.join(save_path, 'stats.csv'), 'w') as f:
            f.write('epoch,name,value\n')
            write_to_file(f, np.argmin(loss), 'loss', np.min(loss))
            write_to_file(f, np.argmin(loss_val), 'val_loss', np.min(loss_val))
            if len(f1) > 0:
                write_to_file(f, np.argmax(f1), 'f1', np.max(f1))
                write_to_file(f, np.argmax(f1_val), 'val_f1', np.max(f1_val))
            write_to_file(f, np.argmax(accuracy), 'accuracy', np.max(accuracy))
            write_to_file(f, np.argmax(accuracy_val), 'val_accuracy', np.max(accuracy_val))


def plot_data(data, title, save_path, y_label):
    for d in data:
        if any(np.isnan(x) for x in d):
            return
        if float('nan') in d:
            return
        plt.plot(d)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path)
    plt.close()


def write_to_file(file, epoch, name, value):
    file.write('{},{},{}\n'.format(epoch, name, value))


if __name__ == '__main__':
    plot()
