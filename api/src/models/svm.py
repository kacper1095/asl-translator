from ..common import initial_environment_config

import numpy as np
import cv2

from ..common.config import Config, DataConfig
from skimage.feature import hog
from skimage import color
from sklearn.svm import SVC
from sklearn.metrics import f1_score


class SvmClassifier(object):
    def __init__(self,
                 C=1.0, kernel='rbf', gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001,
                 cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        self.classifier = SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0, shrinking=shrinking,
                              probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                              verbose=verbose, max_iter=-max_iter, decision_function_shape=decision_function_shape,
                              random_state=None)

    def prepare_image(self, img):
        if Config.BACKEND == 'theano':
            img = img.transpose((1, 2, 0))
        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        img = color.rgb2gray(img)
        features = hog(img, 8, pixels_per_cell=(16, 16), cells_per_block=(10, 10), visualise=False)
        return features

    def one_hot_to_class(self, y):
        return np.argmax(y, axis=1)

    def fit_generator(self, train_generator, samples_per_epoch, nb_epoch, validation_data=None, nb_val_samples=None):
        for epoch in range(nb_epoch):
            i = 0
            print('Epoch: {}/{}'.format(epoch + 1, nb_epoch))
            losses = []
            scores = []
            accuracies = []
            while i < samples_per_epoch:
                x, y = next(train_generator)
                x = self.prepare_x(x)
                y = self.prepare_y(y)

                i += x.shape[0]
                self.classifier.fit(x, y)
                predicted = self.classifier.predict_proba(x)
                losses.append(self.log_loss(y, predicted))
                accuracies.append(self.accuracy(y, predicted))
                scores.append(f1_score(y, np.argmax(predicted, axis=1)))
                print("Loss: {}, accuracy: {}, f1: {}".format(np.mean(losses), np.mean(accuracies), np.mean(val_scores)))

            if validation_data is not None:
                val_loss, val_acc, val_scores = self.validate(validation_data, nb_val_samples)
                print('Val_loss: {}, val_acc: {}, val_f1: {}'.format(val_loss, val_acc, val_scores))

    def validate(self, generator, nb_val_samples):
        losses = []
        accuracies = []
        scores = []
        i = 0
        while i < nb_val_samples:
            x, y = next(generator)
            x = self.prepare_x(x)
            y = self.prepare_y(y)

            i += x.shape[0]
            self.classifier.fit(x, y)
            predicted = self.classifier.predict_proba(x)
            losses.append(self.log_loss(y, predicted))
            accuracies.append(self.accuracy(y, predicted))
            scores.append(f1_score(y, np.argmax(predicted, axis=1)))
        return np.mean(losses), np.mean(accuracies), np.mean(scores)

    def log_loss(self, y_true, y_pred):
        def to_one_hot(data):
            y = np.zeros((data.shape[0], DataConfig.get_number_of_classes()))
            for i, s in enumerate(data):
                y[i, s] = 1.0
            return y
        one_hot_y = to_one_hot(y_true)
        return np.mean(np.sum(one_hot_y * np.log(y_pred + Config.EPSILON), axis=1))

    def accuracy(self, y_true, y_pred):
        return np.mean((y_true == np.round(y_pred)).astype(np.float32))

    def prepare_x(self, x):
            return np.array(list(map(self.prepare_image, x)))

    def prepare_y(self, y):
        return np.array(list(map(self.one_hot_to_class, y)))
