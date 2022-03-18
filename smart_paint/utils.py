import functools

import cv2 as cv
import numpy as np


def image_from_disk(image_path, image_shape=None):
    if not is_image(image_path):
        raise ValueError('"%s" is not an image' % image_path)

    image = cv.imread(image_path)
    if image_shape is not None:
        image = cv.resize(image, tuple(image_shape[:2]))
    return image


def is_image(image_path):
    return any(extension in image_path[-4:] for extension in ['jpg', 'png', 'jpeg'])


def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv.imwrite(out_path, img)


def tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
