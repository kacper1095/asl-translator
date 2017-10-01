import numpy as np
import cv2
import pdb
import _warnings as warnings

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from .perlin_noise import generate_img as generate_perlin_noise

EPSILON = 1e-8

SWITCHES = {
    0: 'elastic_transform',
    1: 'gamma_augmentation',
    2: 'perlin_noise'
}


def get_augmenting_funcions(names):
    def combiner(x):
        for name in names:
            if name not in globals() and name not in SWITCHES.keys():
                warnings.warn("There is no such augmenting function like: " + str(name) +
                              "\nAvailable: " + help())
            else:
                x = globals()[SWITCHES[name] if type(name) == int else name](x)
        return x
    return combiner


def help():
    return "\nelastic_transform" \
           "\ngamma_augmentation" \
            "\nperlin_noise"


def perlin_noise(image):
    return generate_perlin_noise(image)


def elastic_transform(image, alpha=0.15, sigma=0.08, alpha_affine=0.08, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    rescaled = False
    if image.max() < 100:
        image *= 255.
        rescaled = True
    if image.shape[0] < 3:
        image = np.stack((image[0], image[0], image[0]), axis=2)
    shape = image.shape
    alpha *= image.shape[1]
    sigma *= image.shape[1]
    alpha_affine *= image.shape[1]
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    transformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    transformed = transformed[..., 0]
    if rescaled:
        transformed /= 255
    return np.array([transformed])


def gamma_augmentation(x):
    z_value = np.random.uniform(-0.5, 0.5)
    nominator = np.log(0.5 + 2 ** (-0.5) * z_value)
    denominator = np.log(0.5 - 2 ** (-0.5) * z_value)
    gamma_value = nominator/(denominator + EPSILON)
    return x ** gamma_value
