import tensorflow as tf

from image_reader import ImageReader
from defaults import *


def test_image_queue(h=321, w=321):
    input_size = (h, w)

    # Create queue coordinator
    coord = tf.train.Coordinator()

    # Load Image Reader
    with tf.name_scope('create_inputs'):
        reader = ImageReader(
            INDEX_FILE,
            DATA_DIRECTORY,
            MASK_DIRECTORY,
            input_size,
            True,
            True,
            IGNORE_LABEL,
            IMG_MEAN,
            coord)

        image_batch, mask_batch = reader.dequeue(BATCH_SIZE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for _ in range(10):
            images, masks = sess.run([image_batch, mask_batch])
            # img = sess.run(mask_batch)
            print np.unique(masks)

test_image_queue()

