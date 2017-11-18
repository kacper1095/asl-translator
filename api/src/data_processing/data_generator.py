import glob
import cv2
import os
import threading
import numpy as np
import keras.backend as K
import random

random.seed(0)

from ..common import config
from ..common.logger import logger
from keras.preprocessing.image import flip_axis, random_rotation, random_shift, random_zoom
from ..keras_extensions.data_tools.augmenting_tools import gamma_augmentation, poisson_noise, hue_change, brightness_change
from scipy.ndimage.filters import gaussian_filter

from skimage.feature import hog
from skimage import color
from collections import defaultdict
from sklearn.utils import class_weight


def get_images(path):
    files = []
    for ext in ['jpg', 'png', 'jpeg']:
        files.extend(glob.glob(
            os.path.join(path, '**', '*.{}'.format(ext))))
    return files


def get_class_from_path(im_fn):
    components = im_fn.split(os.path.sep)
    return components[-2]


def prepare_image(img):
    img = cv2.resize(img, (config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE))
    img = color.rgb2gray(img)
    features = hog(img, 8, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False)
    return features


def generator(path,
              input_size=64,
              batch_size=32,
              phase=1):
    image_list = np.array(get_images(path))
    print('{} training images in {}'.format(
        image_list.shape[0], path))
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
                image_class = get_class_from_path(im_fn)

                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                new_h, new_w, _ = im.shape
                im = im[:, :, ::-1].astype(np.float32) / 255.
                if phase == config.TrainingConfig.TRAINING_PHASE:
                    im = random_preprocessing(im)
                im = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
                im = np.array([im]).transpose((1, 2, 0))
                if K.backend() == 'tensorflow':
                    images.append(im)
                else:
                    images.append(im.transpose((2, 0, 1)))
                classes.append(config.DataConfig.get_one_hot(image_class))
                if len(images) == batch_size:
                    yield np.array(images), np.array(classes)
                    images = []
                    classes = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                break


def get_generator_for_svm(created_data_generator):
    while True:
        for x, y in created_data_generator:
            y = (y - 1.0) + y
            yield x, y


def generator_with_feature_extraction(path,
                                      input_size=config.Config.IMAGE_SIZE,
                                      batch_size=32,
                                      phase=1):
    image_list = np.array(get_images(path))
    print('{} training images in {}'.format(
        image_list.shape[0], path))
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
                image_class = get_class_from_path(im_fn)

                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                new_h, new_w, _ = im.shape
                im /= 255.
                if phase == config.TrainingConfig.TRAINING_PHASE:
                    im = random_preprocessing(im)
                im = prepare_image(im * 255.)
                images.append(im)
                classes.append(config.DataConfig.get_one_hot(image_class))
                if len(images) == batch_size:
                    yield np.array(images), np.array(classes)
                    images = []
                    classes = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                break


def random_preprocessing(img):
    if random.random() < 0.5:
        img = flip_axis(img, 1)
    img = random_rotation(img, 15, 0, 1, 2)

    img = random_zoom(img, (0.75, 1.25), 0, 1, 2)
    img = random_shift(img, 0.2, 0.2, 0, 1, 2)
    # img = gaussian_filter(img, sigma=[random.uniform(0.0, 0.7), random.uniform(0.0, 0.7),random.uniform(0.0, 0.2)])
    img = poisson_noise(img)
    # img = hue_change(img)
    if random.random() < 0.5:
        img = brightness_change(img)
    else:
        img = gamma_augmentation(img)

    logger.log_img(img[..., ::-1])
    return img


class DataGenerator(object):
    def __init__(self, dir_path, batch_size, input_size, valid, use_hog=False, without_preprocessing=False):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.training_phase = config.TrainingConfig.TESTING_PHASE if valid or without_preprocessing else config.TrainingConfig.TRAINING_PHASE

        if use_hog:
            self.generator = generator_with_feature_extraction(dir_path, config.Config.IMAGE_SIZE,
                                                               batch_size, self.training_phase)
        else:
            self.generator = generator(input_size=input_size, batch_size=batch_size,
                                       path=dir_path, phase=self.training_phase)
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.generator)

    @property
    def number_of_steps(self):
        return len(get_images(self.dir_path)) // self.batch_size

    @property
    def samples_per_epoch(self):
        return len(get_images(self.dir_path))

    @property
    def get_class_weights(self):
        classes = []
        for folder in os.listdir(self.dir_path):
            classes.extend(list(config.DataConfig.get_class(folder)) * len(os.listdir(os.path.join(self.dir_path, folder))))
        class_weights = list(class_weight.compute_class_weight('balanced', np.unique(classes), classes))
        dict_weights = {}
        for i, v in enumerate(class_weights):
            dict_weights[i] = v
        return dict_weights


if __name__ == '__main__':
    pass
