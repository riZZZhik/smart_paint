import os

import tensorflow as tf
from .transform import transform_net
from .utils import save_img, image_from_disk


def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    assert len(paths_out) > 0
    if type(data_in[0]) == str:
        data_in = [image_from_disk(file) for file in data_in]
    img_shape = data_in[0].shape

    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape,
                                                   name='img_placeholder')

        preds = transform_net(img_placeholder)
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        num_iters = int(len(paths_out) / batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos + batch_size]
            X = data_in[pos:pos + batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])

        remaining_in = data_in[num_iters * batch_size:]
        remaining_out = paths_out[num_iters * batch_size:]
        if len(remaining_in) > 0:
            ffwd(remaining_in, remaining_out, checkpoint_dir,
                 device_t=device_t, batch_size=1)
        else:
            return _preds[j]


def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):
    if type(in_path) == str:
        in_path = [in_path]

    ffwd(in_path, [out_path], checkpoint_dir, batch_size=1, device_t=device)
