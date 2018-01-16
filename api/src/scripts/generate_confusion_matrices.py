import matplotlib
matplotlib.use('Agg')

from ..common import initial_environment_config
from ..common.config import TrainingConfig, DataConfig, Config
from sklearn.metrics import confusion_matrix
from ..data_processing.data_generator import DataGenerator

import glob
import itertools
import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sn

from keras.models import load_model

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y']


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(20, 14))
    ax = plt.subplot(111)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    sn.heatmap(cm, annot=True, cmap=cmap, square=True, xticklabels=classes, yticklabels=classes,
               fmt=fmt, ax=ax)
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              verticalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')


class ConfusionMatrixGenerator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.data_gen = DataGenerator(DataConfig.PATHS['VALID_PROCESSED_DATA'], 128, Config.IMAGE_SIZE, True, False)
        self.model = load_model(self.model_path, custom_objects={'f1': lambda x, y: x})

    def generate(self):
        y_true_acc = []
        y_pred_acc = []
        i = 0
        for batch_x, batch_y in tqdm.tqdm(self.data_gen):
            y_pred = self.model.predict(batch_x)
            y_true_acc.append(np.argmax(batch_y, axis=1))
            y_pred_acc.append(np.argmax(y_pred, axis=1))
            i += 1

        y_true = np.concatenate(y_true_acc)
        y_pred = np.concatenate(y_pred_acc)

        y_true = DataConfig.get_letters(y_true)
        y_pred = DataConfig.get_letters(y_pred)

        matrix = confusion_matrix(y_true, y_pred)

        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(matrix, normalize=True, classes=labels)
        plt.savefig(os.path.join(os.path.dirname(self.model_path), 'confusion.png'))


def get_all_models_or_from_path(folder_path):
    if folder_path != '':
        return [os.path.join(folder_path, 'last_weights.h5')]
    weights = []
    for folder in os.listdir(TrainingConfig.PATHS['MODELS']):
        folder_path = os.path.join(TrainingConfig.PATHS['MODELS'], folder)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        for file in files:
            if file == 'weights.h5' or file.endswith('-Best.h5'):
                weights.append(os.path.join(folder_path, file))
    return weights


def confundus():
    model_paths = get_all_models_or_from_path('')
    for model_path in tqdm.tqdm(model_paths):
        try:
            generator = ConfusionMatrixGenerator(model_path)
            generator.generate()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    confundus()
