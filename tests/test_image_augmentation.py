from api.src.keras_extensions.data_tools.augmenting_tools import poisson_noise, gamma_augmentation
from .config import PATHS
from nose import with_setup

import numpy as np
import random
import cv2
import os


def setup():
    random.seed(0)
    np.random.seed(0)


@with_setup(setup)
def test_gamma_augmentation_small_data():
    a = np.array([3, 0, 2])
    new_a = np.array([2.97534947, 0., 1.98961572])
    pred = gamma_augmentation(a)
    assert a.shape == (3,)
    assert all(np.isclose(new_a, pred))


@with_setup(setup)
def test_gamma_augmentation_image():
    test_img = cv2.imread(os.path.join(PATHS['DATA_FOLDER'], 'gamma', 'orig.jpg'))
    test_transormed_img = cv2.imread(os.path.join(PATHS['DATA_FOLDER'], 'gamma', 'augmented.jpg'))

    transformed = gamma_augmentation(test_img / 255.) * 255.
    transformed = transformed.astype(np.uint8)
    assert transformed.shape == test_transormed_img.shape