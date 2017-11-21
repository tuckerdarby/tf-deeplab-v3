import os

import tensorflow as tf


def load_model(loader, sess, ckpt_path):
    checkpoint = tf.train.get_checkpoint_state(ckpt_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        print checkpoint.model_checkpoint_path
        loader.restore(sess, checkpoint.model_checkpoint_path)
        print("Restored model parameters from {}".format(ckpt_path))
    else:
        print 'Did not load model'


def save_model(saver, sess, logdir, global_step):
    checkpoint_file = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, checkpoint_file)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=global_step)
    print('The checkpoint has been created for step {}'.format(sess.run(global_step)))


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)