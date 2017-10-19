import cv2
import numpy as np
import random
import os
import glob

from ..common.config import DataConfig


def process_img(img_file_path, output_img_file_path):
    img = cv2.imread(img_file_path)
    bg = img.copy()
    bg[bg[:, :, -1] > 10, :] = 255.
    bg = bg.astype(np.uint8)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((7, 7), np.uint8)
    where_bg = np.where(bg == 0.0)
    bg[where_bg] == 255.
    eroded_bg = cv2.erode(bg, kernel, iterations=1)
    eroded_bg = cv2.GaussianBlur(eroded_bg, (5, 5), 1)
    where_bg = np.where(eroded_bg < 25)
    img = blend_with_random_background(img, where_bg)
    # img[where_bg] = 0.0
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imwrite(output_img_file_path, img)


def blend_with_random_background(img, where_bg_values):
    mask = np.ones(img.shape, dtype=bool)
    mask[where_bg_values] = False

    backgrounds_filenames = glob.glob(os.path.join(DataConfig.PATHS['RANDOM_BACKGROUNDS_FOLDER'], '*'))
    background = cv2.imread(random.choice(backgrounds_filenames))

    resize_scale = random.choice([0.2, 0.5, 1.0, 1.5, 2.0])
    background = cv2.resize(background, None, fx=resize_scale, fy=resize_scale)
    should_be_rescaled_y = max(img.shape[0], background.shape[0])
    should_be_rescaled_x = max(img.shape[1], background.shape[1])
    background = cv2.resize(background, (should_be_rescaled_x+2, should_be_rescaled_y+2))

    x1 = random.randint(0, background.shape[1] - img.shape[1] - 1)
    y1 = random.randint(0, background.shape[0] - img.shape[0] - 1)
    x2 = x1 + img.shape[1]
    y2 = y1 + img.shape[0]
    cut_bg = background[y1:y2, x1:x2]
    img[~mask] = cut_bg[~mask]
    return img




