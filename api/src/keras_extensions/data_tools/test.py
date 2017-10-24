from api.src.keras_extensions.data_tools.augmenting_tools import gamma_augmentation

import numpy as np
import cv2
np.random.seed(0)

x = (gamma_augmentation(cv2.imread('api/src/keras_extensions/data_tools/orig.jpg') / 255.) * 255.).astype(np.uint8)
cv2.imwrite('augmented.jpg', x)