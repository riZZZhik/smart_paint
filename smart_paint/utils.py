import functools

import cv2 as cv
import numpy as np
from PIL import Image


def image_from_disk(image_path, image_shape=None):
    if not is_image(image_path):
        raise ValueError('"%s" is not an image' % image_path)

    image = Image.open(image_path)
    if image_shape is not None:
        image = image.resize(image_shape[:2])
    image = np.array(image)
    return image


def is_image(image_path):
    return any(type in image_path[-4:] for type in ['jpg', 'png', 'jpeg'])


def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv.imwrite(out_path, img)


def tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
