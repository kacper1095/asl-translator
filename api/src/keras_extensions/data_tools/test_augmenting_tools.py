from .augmenting_tools import get_augmenting_funcions, gamma_augmentation, elastic_transform, perlin_noise

import cv2
import glob
import os
import numpy as np

OUTPUT_PATH = 'output'
INPUT_PATH = 'input'


def main():
    ensure_dirs()
    data = load_data()
    test_elastic(data)
    # test_gamma(data)
    # test_perlin(data)
    # test_all(data)


def ensure_dirs():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(INPUT_PATH):
        os.makedirs(INPUT_PATH)
    files = glob.glob(OUTPUT_PATH + '/*')
    for f in files:
        os.remove(f)


def load_data():
    files = glob.glob('./' + INPUT_PATH + '/*')
    images = []
    for img_file in files:
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(np.array([img]))
    return images


def test_elastic(data):
    for i, img in enumerate(data):
        out_img = elastic_transform(img/255.)
        cv2.imwrite(os.path.join(OUTPUT_PATH, str(i) + '_elastic.png'), out_img[0]*255.)


def test_gamma(data):
    for i, img in enumerate(data):
        out_img = gamma_augmentation(img/255.)
        cv2.imwrite(os.path.join(OUTPUT_PATH, str(i) + '_gamma.png'), out_img[0]*255.)


def test_perlin(data):
    for i, img in enumerate(data):
        out_img = perlin_noise(img/255.)
        cv2.imwrite(os.path.join(OUTPUT_PATH, str(i) + '_perlin.png'), out_img[0] * 255.)


def test_all(data):
    for i, img in enumerate(data):
        out_img = get_augmenting_funcions([0, 1])(img/255.)
        cv2.imwrite(os.path.join(OUTPUT_PATH, str(i) + '_all.png'), out_img[0] * 255.)

if __name__ == '__main__':
    main()
