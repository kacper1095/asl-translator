from keras.callbacks import Callback
from ..common.config import Config

import numpy as np
import os
import cv2


class DifficultSamplesSaver(Callback):
    def __init__(self, save_path, valid_data):
        super().__init__()
        self.save_path = save_path
        self.valid_data = valid_data

    def on_train_begin(self, logs=None):
        if not os.path.exists(os.path.join(self.save_path)):
            os.makedirs(os.path.join(self.save_path))

    def on_train_end(self, logs=None):
        samples = 0
        for batch_x, batch_y in self.valid_data:
            predicted = self.model.predict_on_batch(batch_x)
            y_true = np.argmax(batch_y, axis=1)
            y_pred = np.argmax(predicted, axis=1)
            missclassified = np.where(y_true != y_pred)[0]
            difficult = batch_x[missclassified]
            nb_files = len(os.listdir(self.save_path))
            for sample in difficult:
                cv2.imwrite(os.path.join(self.save_path, '%d.png' % nb_files),
                            sample.transpose((1, 2, 0)) * 255)
                nb_files += 1
            samples += len(batch_x)
            if samples >= self.valid_data.samples_per_epoch:
                break


