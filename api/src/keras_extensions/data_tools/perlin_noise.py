import math
import numpy as np
import random

random.seed(0)

image_size = 256

p = np.array(
    [215, 143, 190, 43, 73, 139, 87, 131, 35, 50, 56, 180, 224, 144, 39, 218, 120, 37, 173, 222, 156, 71, 239, 210, 151,
     142, 158, 64, 95, 60, 83, 93, 214, 148, 10, 119, 80, 58, 205, 29, 213, 135, 197, 183, 2, 242, 53, 96, 111, 0, 7,
     25, 168, 63, 91, 216, 233, 54, 179, 109, 85, 104, 140, 255, 124, 134, 184, 6, 13, 227, 160, 40, 159, 212, 90, 217,
     195, 244, 250, 3, 24, 101, 47, 162, 220, 185, 11, 48, 106, 141, 157, 68, 204, 77, 128, 167, 165, 55, 161, 118, 240,
     252, 115, 16, 12, 42, 62, 41, 203, 226, 19, 230, 121, 238, 149, 22, 241, 74, 132, 136, 23, 170, 254, 228, 89, 52,
     147, 208, 234, 236, 30, 36, 75, 34, 76, 201, 193, 129, 154, 122, 237, 28, 31, 249, 14, 253, 69, 100, 8, 127, 133,
     146, 130, 246, 137, 251, 17, 18, 107, 231, 126, 113, 86, 164, 81, 174, 108, 88, 79, 189, 15, 243, 196, 248, 206,
     66, 207, 103, 177, 171, 114, 198, 153, 176, 182, 221, 166, 209, 82, 186, 199, 20, 163, 94, 125, 72, 97, 1, 211, 99,
     219, 191, 145, 67, 202, 49, 169, 175, 33, 44, 245, 223, 117, 150, 92, 21, 247, 155, 235, 65, 32, 172, 26, 194, 178,
     225, 9, 152, 4, 5, 181, 188, 27, 78, 38, 46, 105, 187, 200, 61, 102, 51, 110, 57, 59, 70, 192, 138, 123, 98, 45,
     116, 232, 229, 84, 112])


def generate_img(img, cube_length=7.0):
    global p

    img_min = img.min()
    img_max = img.max()

    permuation = np.random.permutation(len(p))
    p = p[permuation]
    c, h, w = img.shape
    array = np.ones((h, w), dtype=np.float32) * img.max()
    # xs, ys = np.meshgrid(w, h)
    # values = sum_of_perlins(xs, ys)
    # noise = array * values
    for y in range(h):
        for x in range(w):
            array[y, x] *= sum_of_perlins(x, y, cube_length)

    noise = np.clip(array, img_min - 0.1 * img_min, img_max)
    alpha = random.uniform(0.4, 0.8)
    final_img = alpha * img + (1 - alpha) * noise
    return final_img


@np.vectorize
def sum_of_perlins(x, y, cube_length):
    value = perlin(x, y, cube_length)
    # value = abs(perlin(x, y))
    for i in range(2, 10, 2):
        # value += abs(1 / i * perlin(i * x, y * i))
        value += 1 / i * perlin(i * x, y * i, cube_length)
    return value


def perlin(x, y, cube_length):
    x /= cube_length
    y /= cube_length
    x0 = math.floor(x)
    x1 = int(x0 + 1)
    y0 = math.floor(y)
    y1 = int(y0 + 1)
    sx = x - x0
    sy = y - y0
    sx = fade(sx)
    sy = fade(sy)
    n0 = dot_grid_gradient(x0, y0, x, y)
    n1 = dot_grid_gradient(x1, y0, x, y)
    ix0 = linear_interpolation(n0, n1, sx)
    n0 = dot_grid_gradient(x0, y1, x, y)
    n1 = dot_grid_gradient(x1, y1, x, y)
    ix1 = linear_interpolation(n0, n1, sx)
    value = linear_interpolation(ix0, ix1, sy)
    return value


def fade(t):
    return 3 * t ** 2 - 2 * t ** 3


def grad(p, x, y, z=0):
    switch = p & 0xF
    if switch == 0x0:
        return x + y
    elif switch == 0x1:
        return -x + y
    elif switch == 0x2:
        return x - y
    elif switch == 0x3:
        return -x - y
    elif switch == 0x4:
        return x + z
    elif switch == 0x5:
        return -x + z
    elif switch == 0x6:
        return x - z
    elif switch == 0x7:
        return -x - z
    elif switch == 0x8:
        return y + z
    elif switch == 0x9:
        return -y + z
    elif switch == 0xA:
        return y - z
    elif switch == 0xB:
        return -y - z
    elif switch == 0xC:
        return y + x
    elif switch == 0xD:
        return -y + x
    elif switch == 0xE:
        return y - x
    elif switch == 0xF:
        return -y - x
    return 0


def dot_grid_gradient(ix, iy, x, y, iz=0):
    dx = x - float(ix)
    dy = y - float(iy)
    p_value = p[(ix + p[(iy + p[iz]) % 256]) % 256]
    gradient = grad(p_value, dx, dy)
    return gradient


def linear_interpolation(a0, a1, w):
    return (1.0 - w) * a0 + w * a1


