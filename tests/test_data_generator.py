from api.src.data_processing.data_generator import DataGenerator
from PIL import Image

import os
import numpy as np


def test_get_images_blank():
    data_gen = DataGenerator(os.path.join('data', 'bland'), 1, 64, False)
    assert data_gen.get_images() == []


def test_get_images():
    true_values = ['data{0}letters{0}a{0}augmented.jpg'.format(os.path.sep), 'data{0}letters{0}a{0}orig.jpg'.format(os.path.sep),
                   'data{0}letters{0}b{0}augmented.jpg'.format(os.path.sep), 'data{0}letters{0}b{0}orig.jpg'.format(os.path.sep)]
    data_letters = os.path.join('data', 'letters')
    data_gen = DataGenerator(data_letters, 1, 64, False)
    assert all(img in true_values for img in data_gen.get_images()) and len(
        data_gen.get_images()) == len(true_values)


def test_get_class_from_filename():
    data_gen = DataGenerator(os.path.join('data', 'letters'), 1, 64, False)
    paths = data_gen.get_images()
    true_classes = sorted(['a', 'a', 'b', 'b'])
    tested = sorted([data_gen.get_class_from_path(p) for p in paths])
    assert all(x == y for x, y in zip(true_classes, tested))


def test_prepare_image():
    data_gen = DataGenerator(None, 32, 64, False)
    img_w = img_h = 64
    bias = np.random.rand(img_w, img_h, 1) * 64
    variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
    imarray = np.random.rand(img_w, img_h, 3) * variance + bias
    im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    im = np.asarray(im)
    features = data_gen.prepare_image(im)
    assert len(features) == 2592
    assert len(features.shape) == 1
    assert features.shape == (2592,)
    assert all(0.0 < val < 1.0 for val in features)
