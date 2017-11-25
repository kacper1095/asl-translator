import glob
import cv2
import os
import threading
import numpy as np
import keras.backend as K
import random

random.seed(0)

from ..common.config import DataConfig, TrainingConfig, Config
from ..common.logger import logger
from keras.preprocessing.image import flip_axis, random_rotation, random_shift, random_zoom
from ..keras_extensions.data_tools.augmenting_tools import gamma_augmentation, poisson_noise, hue_change, \
    brightness_change

from skimage.feature import hog
from skimage import color
from sklearn.utils import class_weight


class DataGenerator(object):
    def __init__(self, dir_path, batch_size, input_size, valid, use_hog=False, without_preprocessing=False):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.training_phase = TrainingConfig.TESTING_PHASE if valid or without_preprocessing else TrainingConfig.TRAINING_PHASE

        if use_hog:
            self.generator = self.generator_with_feature_extraction(self.training_phase)
        else:
            self.generator = self.generator(self.training_phase)
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.generator)

    @property
    def number_of_steps(self):
        return len(self.get_images()) // self.batch_size

    @property
    def samples_per_epoch(self):
        return len(self.get_images())

    @property
    def get_class_weights(self):
        classes = []
        for folder in os.listdir(self.dir_path):
            classes.extend(list(DataConfig.get_class(folder)) * len(os.listdir(os.path.join(self.dir_path, folder))))
        class_weights = list(class_weight.compute_class_weight('balanced', np.unique(classes), classes))
        dict_weights = {}
        for i, v in enumerate(class_weights):
            dict_weights[i] = v
        return dict_weights

    def generator(self,
                  phase=1):
        image_list = np.array(self.get_images())
        print('{} training images in {}'.format(
            image_list.shape[0], self.dir_path))
        index = np.arange(0, image_list.shape[0])

        while True:
            images = []
            classes = []
            np.random.shuffle(index)
            for i in index:
                try:
                    im_fn = image_list[i]
                    im = cv2.imread(im_fn)
                    # print im_fn
                    h, w, _ = im.shape
                    image_class = self.__get_class_from_path(im_fn)

                    resize_h = self.input_size
                    resize_w = self.input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    new_h, new_w, _ = im.shape
                    im = im[:, :, ::-1].astype(np.float32) / 255.
                    if phase == TrainingConfig.TRAINING_PHASE:
                        im = self.__random_preprocessing(im)
                    if K.backend() == 'tensorflow':
                        images.append(im)
                    else:
                        images.append(im.transpose((2, 0, 1)))
                    classes.append(DataConfig.get_one_hot(image_class))
                    if len(images) == self.batch_size:
                        yield np.array(images), np.array(classes)
                        images = []
                        classes = []
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    break

    def generator_with_feature_extraction(self, phase=1):
        image_list = np.array(self.get_images())
        print('{} training images in {}'.format(
            image_list.shape[0], self.dir_path))
        index = np.arange(0, image_list.shape[0])
        while True:
            images = []
            classes = []
            np.random.shuffle(index)
            for i in index:
                try:
                    im_fn = image_list[i]
                    im = cv2.imread(im_fn).astype(np.float32)
                    # print im_fn
                    h, w, _ = im.shape
                    image_class = self.__get_class_from_path(im_fn)

                    resize_h = self.input_size
                    resize_w = self.input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    new_h, new_w, _ = im.shape
                    if phase == TrainingConfig.TRAINING_PHASE:
                        im = self.__random_preprocessing(im)
                    im = self.prepare_image(im)
                    images.append(im)
                    classes.append(DataConfig.get_one_hot(image_class))
                    if len(images) == self.batch_size:
                        yield np.array(images), np.array(classes)
                        images = []
                        classes = []
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    break

    def prepare_image(self, img, cells_per_block=(3, 3)):
        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        img = color.rgb2gray(img)
        features = hog(img, 8, pixels_per_cell=(8, 8), cells_per_block=cells_per_block, visualise=False)
        return features

    def get_images(self):
        files = []
        for ext in ['jpg', 'png', 'jpeg']:
            files.extend(glob.glob(
                os.path.join(self.dir_path, '**', '*.{}'.format(ext))))
        return files

    def get_class_from_path(self, im_fn):
        components = im_fn.split(os.path.sep)
        return components[-2]

    def __random_preprocessing(self, img):
        if random.random() < 0.5:
            img = flip_axis(img, 1)
        img = random_rotation(img, 7.5, 0, 1, 2)

        img = random_zoom(img, (0.75, 1.25), 0, 1, 2)
        img = random_shift(img, 0.2, 0.2, 0, 1, 2)
        # img = gaussian_filter(img, sigma=[random.uniform(0.0, 0.7), random.uniform(0.0, 0.7),random.uniform(0.0, 0.2)])
        img = poisson_noise(img)
        img = hue_change(img)
        if random.random() < 0.5:
            img = brightness_change(img)
        else:
            img = gamma_augmentation(img)

        logger.log_img(img[..., ::-1])
        return img


if __name__ == '__main__':
    pass
