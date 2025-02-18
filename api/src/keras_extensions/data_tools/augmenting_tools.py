import numpy as np
import cv2
import pdb
import _warnings as warnings
import scipy.ndimage as ndi

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

EPSILON = 1e-8

SWITCHES = {
    0: 'elastic_transform',
    1: 'gamma_augmentation',
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



def elastic_transform(image, alpha=0.15, sigma=0.08, alpha_affine=0.08, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    :param image: Image to transform in range [0, 1] or [0, 255].
    :param alpha: Size of random elastic transform (distance of moving pixels).
    :param sigma: Standard deviation for gaussian blur (smoothing coefficient).
    :param alpha_affine: Coefficient for translating image.
    :param random_state: State of random number generator.
    :return: Warped image in the same range as input.
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
    z_value = np.random.uniform(-0.25, 0.25)
    nominator = np.log(0.5 + 2 ** (-0.5) * z_value)
    denominator = np.log(0.5 - 2 ** (-0.5) * z_value)
    gamma_value = nominator / (denominator + EPSILON)
    return x ** gamma_value


def poisson_noise(x):
    peak = np.random.uniform(0.95, 1.0)
    noisy = np.random.poisson(x * 255.0 * peak) / peak / 255.0
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


def brightness_change(x):
    x = cv2.cvtColor((x * 255).astype(np.float32), cv2.COLOR_RGB2HSV)
    random_bright = .5 + np.random.random()
    x[:, :, 2] *= random_bright
    x[:, :, 2] = np.clip(x[:, :, 2], 0, 255)
    x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
    return x / 255.


def hue_change(x):
    x = cv2.cvtColor((x * 255).astype(np.float32), cv2.COLOR_RGB2HSV)
    random_hue = np.random.uniform(-5, 5)
    x[:, :, 0] += random_hue
    x[:, :, 0] = x[:, :, 0] % 360
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    return x / 255.


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """
    Performs a random spatial shift of a Numpy image tensor.
    :param x: Input tensor. Must be 3D.
    :param wrg: Width shift range, as a float fraction of the width.
    :param hrg: Height shift range, as a float fraction of the height.
    :param row_axis: Index of axis for rows in the input tensor.
    :param col_axis: Index of axis for columns in the input tensor.
    :param channel_axis: Index of axis for channels in the input tensor.
    :param fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    :return: Randomly shifted Numpy tensor
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel
                      in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """
    Performs a random spatial zoom of a Numpy image tensor.
    :param x: Input tensor. Must be 3D.
    :param zoom_range: Tuple of floats; zoom range for width and height.
    :param row_axis: Index of axis for rows in the input tensor.
    :param col_axis: Index of axis for columns in the input tensor.
    :param channel_axis: Index of axis for channels in the input tensor.
    :param fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    :return: Zoomed Numpy image tensor.
    :raise: ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
