from api.src.keras_extensions.data_tools.augmenting_tools import poisson_noise, gamma_augmentation, brightness_change, hue_change, random_shift, random_zoom
from PIL import Image

import numpy as np


class TestAugmentingTools(object):

    def setup(cls):
        cls.img_w = cls.img_h = 20
        rgb_images = []
        for n in range(8):
            bias = np.random.rand(cls.img_w, cls.img_h, 1) * 64
            variance = np.random.rand(cls.img_w, cls.img_h, 1) * (255 - 64)
            imarray = np.random.rand(cls.img_w, cls.img_h, 3) * variance + bias
            im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
            rgb_images.append(np.asarray(im))

        cls.all_test_images = np.array(rgb_images).astype(np.float32) / 255.

    def teardown(cls):
        del cls.all_test_images

    def test_gamma_augmentation(self):
        for img in self.all_test_images:
            augmented = gamma_augmentation(img)
            assert augmented.shape == (20, 20, 3)
            assert augmented.min() >= 0.0 and augmented.max() <= 1.0

    def test_poison_noise(self):
        for img in self.all_test_images:
            augmented = poisson_noise(img)
            assert augmented.shape == (20, 20, 3)
            assert augmented.min() >= 0.0 and augmented.max() <= 1.0

    def test_hue_change(self):
        for img in self.all_test_images:
            augmented = hue_change(img)
            assert augmented.shape == (20, 20, 3)
            assert augmented.min() >= 0.0 and augmented.max() <= 1.0

    def test_brightness_change(self):
        for img in self.all_test_images:
            augmented = brightness_change(img)
            assert augmented.shape == (20, 20, 3)
            assert augmented.min() >= 0.0 and augmented.max() <= 1.0

    def test_zoom_change(self):
        for img in self.all_test_images:
            augmented = random_zoom(img, (0.25, 20))
            assert augmented.shape == (20, 20, 3)
            assert augmented.min() >= 0.0 and augmented.max() <= 1.0

    def test_shift_change(self):
        for img in self.all_test_images:
            augmented = random_shift(img, 10, 10)
            assert augmented.shape == (20, 20, 3)
            assert augmented.min() >= 0.0 and augmented.max() <= 1.0