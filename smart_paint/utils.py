import numpy as np
from PIL import Image


def image_from_disk(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    return image
