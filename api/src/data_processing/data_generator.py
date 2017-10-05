import glob
import cv2
import os
import threading
import numpy as np
import keras.backend as K
import random

random.seed(0)

from ..common import config
from keras.preprocessing.image import flip_axis, random_rotation


def get_images(path):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(path, '**', '*.{}'.format(ext))))
    return files


def get_class_from_path(im_fn):
    components = im_fn.split(os.path.sep)
    return components[-2]


def crop_area(im, max_tries=50):
    """
    make random crop from the input image
    :param im:
    :param max_tries:
    :return:
    """
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        return im

    return im


def generator(path,
              input_size=64,
              batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 0.75, 1, 1.5, 2.0])):
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

                rd_scale = np.random.choice(random_scale)
                # im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                # # print rd_scale
                # # random crop a area from image
                # if np.random.rand() < background_ratio:
                #     # crop background
                #     im = crop_area(im)
                #     # pad and resize image
                #     new_h, new_w, _ = im.shape
                #     max_h_w_i = np.max([new_h, new_w, input_size])
                #     im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                #     im_padded[:new_h, :new_w, :] = im.copy()
                #     im = cv2.resize(im_padded, dsize=(input_size, input_size))
                # else:
                #     im = crop_area(im)
                #     h, w, _ = im.shape
                #
                #     # pad the image to the training input size or the longer side of image
                #     new_h, new_w, _ = im.shape
                #     max_h_w_i = np.max([new_h, new_w, input_size])
                #     im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                #     im_padded[:new_h, :new_w, :] = im.copy()
                #     im = im_padded
                #     # resize the image to input size
                #     new_h, new_w, _ = im.shape
                #     resize_h = input_size
                #     resize_w = input_size
                #     im = cv2.resize(im, dsize=(resize_w, resize_h))
                #     new_h, new_w, _ = im.shape

                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                new_h, new_w, _ = im.shape

                im = im[:, :, ::-1].astype(np.float32)
                im = random_preprocessing(im)
                if K.backend() == 'tensorflow':
                    images.append(im)
                else:
                    images.append(im.transpose((2, 0, 1)))
                classes.append(config.DataConfig.get_one_hot(image_class))
                if len(images) == batch_size:
                    yield np.array(images) / 255., np.array(classes)
                    images = []
                    classes = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                break


def random_preprocessing(img):
    if random.random() < 0.5:
        img = flip_axis(img, 2)
    if random.random() < 0.5:
        img = random_rotation(img, random.uniform(-5, 5), 0, 1, 2, fill_mode='wrap')
    return img


class DataGenerator(object):
    def __init__(self, dir_path, batch_size, input_size):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.generator = generator(input_size=input_size, batch_size=batch_size, path=dir_path)
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


if __name__ == '__main__':
    pass

