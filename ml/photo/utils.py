import os

import scipy.misc
import numpy as np
from PIL import Image


def is_image(filename):
    return any(t.lower() in filename[-4:] for t in ['jpg', 'png', 'jpeg'])


def save_img(out_path, img):
    # TODO: If image pixels are floats 0-1 transform to 0-256 uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(out_path)


def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1 = Image.open(style_path).size
    o2 = 3  # FIXME

    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = get_img(style_path, img_size=new_shape)
    return style_target


def get_img(img, img_size=None):
    if type(img) == str:
        img = Image.open(img)

    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))

    if img_size:
        img = scipy.misc.imresize(img, img_size)
    return img


def exists(p, msg):
    assert os.path.exists(p), msg


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files


def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]
