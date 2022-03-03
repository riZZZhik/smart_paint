import functools
import os
import time

import numpy as np
import transform
import tensorflow as tf
from loguru import logger

from . import vgg
from .evaluate import ffwd_to_img
from .utils import image_from_disk, is_image, tensor_size


class SmartPaintTrain:
    def __init__(self, vgg_path, style_target):
        # Check input variables
        self.vgg_path = vgg_path
        self.style_target = image_from_disk(style_target) if type(style_target) is str else style_target

        # Build Network
        self.style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
        self.content_layer = 'relu4_2'
        self.net, self.style_features = self.__build_network(vgg_path, self.style_target, self.style_layers)

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

        return net, style_features

    def train(self, test_img, test_dir, epochs, batch_size, content_targets, checkpoint_dir, print_iterations,
              weights=None, learning_rate=1e-3):
        # Check input variables
        if weights is None:
            weights = {
                'content': 7.5e0,
                'style': 1e2,
                'tv': 2e2
            }
        elif not {"content", "style", "tv"}.issubset(set(weights.keys())):
            raise ValueError('Weight dictionary must have "content", "style" and "tv" keys')

        if os.path.isdir(content_targets):
            content_targets = [os.path.join(content_targets, image)
                               for image in os.listdir(content_targets) if is_image(image)]

        # Run training
        for preds, losses, i, epoch in self._optimize(epochs, batch_size, content_targets, checkpoint_dir,
                                                      weights, learning_rate, print_iterations):
            # Log losses
            style_loss, content_loss, tv_loss, loss = losses
            logger.info('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
            logger.info('Style: %s, Content:%s, Tv: %s' % (style_loss, content_loss, tv_loss))

            # Save evaluation
            save_path = '%s/%s_%s.png' % (test_dir, epoch, i)
            ffwd_to_img(test_img, save_path, checkpoint_dir)

    def _optimize(self, epochs, batch_size, content_targets, checkpoint_dir,
                  weights, learning_rate, print_iterations):

        # Check batch_size
        batch_shape = (batch_size, 256, 256, 3)
        mod = len(content_targets) % batch_size
        if mod > 0:
            print("Train set has been trimmed slightly..")
            content_targets = content_targets[:-mod]

        with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
            X_content = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name="X_content")
            X_pre = vgg.preprocess(X_content)

            # precompute content features
            content_features = {}
            content_net = vgg.vgg(self.vgg_path, X_pre)
            content_features[self.content_layer] = content_net[self.content_layer]

            preds = transform.net(X_content / 255.0)
            preds_pre = vgg.preprocess(preds)
            net = vgg.vgg(self.vgg_path, preds_pre)

            content_size = tensor_size(content_features[self.content_layer]) * batch_size
            assert tensor_size(content_features[self.content_layer]) == tensor_size(net[self.content_layer])
            content_loss = weights["content"] * (2 * tf.nn.l2_loss(
                net[self.content_layer] - content_features[self.content_layer]) / content_size
                                                 )

            style_losses = []
            for style_layer in self.style_layers:
                layer = net[style_layer]
                bs, height, width, filters = map(lambda i: i, layer.get_shape())
                size = height * width * filters
                feats = tf.reshape(layer, (bs, height * width, filters))
                feats_T = tf.transpose(a=feats, perm=[0, 2, 1])
                grams = tf.matmul(feats_T, feats) / size
                style_gram = self.style_features[style_layer]
                style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)

            style_loss = weights["style"] * functools.reduce(tf.add, style_losses) / batch_size

            # total variation denoising
            tv_y_size = tensor_size(preds[:, 1:, :, :])
            tv_x_size = tensor_size(preds[:, :, 1:, :])
            y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1] - 1, :, :])
            x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2] - 1, :])
            tv_loss = weights["tv"] * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

            loss = content_loss + style_loss + tv_loss

            # overall loss
            train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
            sess.run(tf.compat.v1.global_variables_initializer())
            import random
            uid = random.randint(1, 100)
            logger.info("UID: %s" % uid)
            for epoch in range(epochs):
                num_examples = len(content_targets)
                iterations = 0
                while iterations * batch_size < num_examples:
                    start_time = time.time()
                    curr = iterations * batch_size
                    step = curr + batch_size
                    X_batch = np.zeros(batch_shape, dtype=np.float32)
                    for j, img_p in enumerate(content_targets[curr:step]):
                        X_batch[j] = image_from_disk(img_p, (256, 256, 3)).astype(np.float32)

                    iterations += 1
                    assert X_batch.shape[0] == batch_size

                    feed_dict = {
                        X_content: X_batch
                    }

                    train_step.run(feed_dict=feed_dict)
                    end_time = time.time()
                    delta_time = end_time - start_time
                    is_print_iter = int(iterations) % print_iterations == 0
                    is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                    should_print = is_print_iter or is_last
                    if should_print:
                        to_get = [style_loss, content_loss, tv_loss, loss, preds]
                        test_feed_dict = {
                            X_content: X_batch
                        }

                        tup = sess.run(to_get, feed_dict=test_feed_dict)
                        _style_loss, _content_loss, _tv_loss, _loss, _preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _loss)
                        saver = tf.compat.v1.train.Saver()
                        res = saver.save(sess, checkpoint_dir)
                        yield preds, losses, iterations, epoch
