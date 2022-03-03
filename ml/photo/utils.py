import os
import numpy as np
import scipy.misc
import sys

from cv2 import imread, imwrite


def isimage(str):
  return any(type in str[-4:] for type in ['jpg', 'png', 'jpeg'])

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imwrite(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(img, img_size=False):
   if type(img) == str:
     img = imread(img)
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
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
    return [os.path.join(img_dir,x) for x in files]