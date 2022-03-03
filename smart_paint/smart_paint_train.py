import numpy as np
import tensorflow as tf
from loguru import logger

import vgg
from utils import image_from_disk


class SmartPaintTrain:
    def __init__(self, vgg_path, style_target):
        # Check input variables
        self.style_target = image_from_disk(style_target) if type(style_target) is str else style_target

        # Build Network
        style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
        self.net = self.__build_network(vgg_path, self.style_target, style_layers)

        logger.info("Successfully built Smart Paint Training network")

    @staticmethod
    def __build_network(vgg_path, style_target, style_layers):
        # Check input variables
        if type(style_target) is str:
            style_target = image_from_disk(style_target)

        style_features = {}
        style_shape = (1,) + style_target.shape

        # Precompute style features
        with tf.Graph().as_default(), tf.device('/cpu:0'), tf.compat.v1.Session() as sess:
            style_image = tf.compat.v1.placeholder(tf.float32, shape=style_shape, name='style_image')
            style_image_pre = vgg.preprocess(style_image)
            net = vgg.vgg(vgg_path, style_image_pre)
            style_pre = np.array([style_target])
            for layer in style_layers:
                features = net[layer].eval(feed_dict={style_image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[layer] = gram

        return net

    def train(self, batch_size, content_targets):
        mod = len(content_targets) % batch_size
        if mod > 0:
            print("Train set has been trimmed slightly..")
            content_targets = content_targets[:-mod]
