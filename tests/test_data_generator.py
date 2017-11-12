from api.src.data_processing import data_generator
from api.src.common.config import DataConfig
from PIL import Image

import os
import numpy as np


def test_get_images_blank():
    assert data_generator.get_images(os.path.join('data', 'bland')) == []


def test_get_images():
    true_values = ['data{0}letters{0}a{0}augmented.jpg'.format(os.path.sep), 'data{0}letters{0}a{0}orig.jpg'.format(os.path.sep),
                   'data{0}letters{0}b{0}augmented.jpg'.format(os.path.sep), 'data{0}letters{0}b{0}orig.jpg'.format(os.path.sep)]
    data_letters = os.path.join('data', 'letters')
    assert all(img in true_values for img in data_generator.get_images(data_letters)) and len(
        data_generator.get_images(data_letters)) == len(true_values)


def test_get_class_from_filename():
    paths = data_generator.get_images(os.path.join('data', 'letters'))
    true_classes = ['a', 'a', 'b', 'b']
    tested = [data_generator.get_class_from_path(p) for p in paths]
    assert all(x == y for x, y in zip(true_classes, tested))


def test_prepare_image():
    img_w = img_h = 64
    bias = np.random.rand(img_w, img_h, 1) * 64
    variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
    imarray = np.random.rand(img_w, img_h, 3) * variance + bias
    im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    im = np.asarray(im)
    features = data_generator.prepare_image(im)
    assert len(features) == 2592
    assert len(features.shape) == 1
    assert features.shape == (2592,)
    assert all(0.0 < val < 1.0 for val in features)
