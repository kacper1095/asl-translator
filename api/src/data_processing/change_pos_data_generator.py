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
from ..keras_extensions.data_tools.augmenting_tools import gamma_augmentation, poisson_noise, brightness_change
from sklearn.utils import class_weight
from collections import defaultdict


def get_class_from_path(im_fn):
    components = im_fn.split(os.path.sep)
    return components[-2]


def random_preprocessing(img):
    # if random.random() < 0.5:
    #     img = flip_axis(img, 1)
    img = random_rotation(img, 10, 0, 1, 2)

    img = random_zoom(img, (0.75, 1.3), 0, 1, 2)
    img = random_shift(img, 0.25, 0.25, 0, 1, 2)
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
    def __init__(self, dir_path, batch_size, input_size, valid):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.training_phase = config.TrainingConfig.TESTING_PHASE if valid else config.TrainingConfig.TRAINING_PHASE

        self.generator = self.generator()
        self.lock = threading.Lock()
        self.samples_per_class_per_char = 2000

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.generator)

    @property
    def number_of_steps(self):
        return len(self.get_negative_positive_pairs()) // self.batch_size

    @property
    def samples_per_epoch(self):
        return len(self.get_negative_positive_pairs())

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

    def get_images(self):
        files = defaultdict(list)
        for char in config.DataConfig.AVAILABLE_CHARS:
            for ext in ['jpg', 'png', 'jpeg']:
                files[char].extend(glob.glob(
                    os.path.join(self.dir_path, char, '*.{}'.format(ext))))
        return files

    def generator(self):
        image_list = np.array(self.get_negative_positive_pairs())
        index = np.arange(0, image_list.shape[0])

        while True:
            images_anchors = []
            images_references = []
            classes = []
            np.random.shuffle(index)
            for i in index:
                try:
                    pair, pair_class = image_list[i]
                    first_image = self.__preprocess_img_from_path(pair[0], is_positive=pair_class)
                    second_image = self.__preprocess_img_from_path(pair[1], is_positive=pair_class)

                    images_anchors.append(first_image)
                    images_references.append(second_image)

                    classes.append([0.0, 1.0] if pair_class else [1.0, 0.0])
                    if len(images_anchors) > self.batch_size:
                        yield [np.array(images_anchors), np.array(images_references)], np.array(classes)
                        images_anchors.clear()
                        images_references.clear()
                        classes = []
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    break

    def __preprocess_img_from_path(self, img_path, is_positive=False):
        im = cv2.imread(img_path)
        # print im_fn
        h, w, _ = im.shape

        resize_h = self.input_size
        resize_w = self.input_size
        im = cv2.resize(im, dsize=(resize_w, resize_h))
        new_h, new_w, _ = im.shape
        im = im[:, :, ::-1].astype(np.float32) / 255.
        if self.training_phase == config.TrainingConfig.TRAINING_PHASE and is_positive:
            im = random_preprocessing(im)
        if K.backend() == 'tensorflow':
            return im
        else:
            return im.transpose((2, 0, 1))

    def get_negative_positive_pairs(self):
        negative_class = 0
        positive_class = 1
        available_classes = config.DataConfig.AVAILABLE_CHARS
        images_paths = self.get_images()
        pairs = []
        i = 0
        for char in available_classes:
            for key, items in images_paths.items():
                for _ in range(self.samples_per_class_per_char // (len(available_classes) - 1)):
                    anchor = random.choice(images_paths[char])
                    negative = random.choice(images_paths[key])
                    pairs.append(((anchor, negative), positive_class))
                    i += 1
            for _ in range(self.samples_per_class_per_char):
                anchor = random.choice(images_paths[char])
                positive = anchor
                pairs.append(((anchor, positive), negative_class))
        print('Generated: {} negative samples'.format(i))
        print('Generated: {} positive samples'.format(self.samples_per_class_per_char * len(available_classes)))
        print('Generated: {} samples in total'.format(len(pairs)))
        return pairs


if __name__ == '__main__':
    pass
