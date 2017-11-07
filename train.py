from __future__ import division

import argparse
import os
import time

import tensorflow as tf

from net import DeepLabResNetModel
from image_reader import ImageReader
from utils import decode_labels, prepare_labels, inv_preprocess
from defaults import *


def get_arguments():
    parser = argparse.ArgumentParser(description='DeepLab-ResNet Network')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Number of images per batch')
    parser.add_argument('--index-loc', type=str, default=INDEX_FILE,
                        help='Path to the file containing the matching files')
    parser.add_argument('--data-dir', type=str, default=DATA_DIRECTORY,
                        help='Path to the directory containing the training data')
    parser.add_argument('--mask-dir', type=str, default=MASK_DIRECTORY,
                        help='Path to the directory containing the masks of training data')
    parser.add_argument('--ignore-label', type=str, default=IGNORE_LABEL,
                        help='The index of the label to ignore during training')
    parser.add_argument('--input-size', type=str, default=INPUT_SIZE,
                        help='Comma-separated string with height and width of images')
    parser.add_argument('--is-training', action='store_true',
                        help='Whether to update the running means and variances during training')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help='Base learning rate for training with polynomial decay')
    parser.add_argument('--momentum', type=float, default=MOMENTUM,
                        help='Momentum component of the optimizer')
    parser.add_argument('--not-restore-last', action='store_true',
                        help='Whether to not restore last FC layers')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                        help='Number of classes to predict, including background')
    parser.add_argument('--num-steps', type=int, default=NUM_STEPS,
                        help='Number of training steps')
    parser.add_argument('--power', type=float, default=POWER,
                        help='Decay parameter to compute the learning rate')
    parser.add_argument('--random-mirror', action='store_true',
                        help='Whether to randomly mirror the inputs during the training')
    parser.add_argument('--random-scale', action='store_true',
                        help='Whether to randomly scale the inputs during the training')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='Random seed to have reproducible results')
    parser.add_argument('--restore-from', type=str, default=RESTORE_FROM,
                        help='Where restore model parameters are from')
    parser.add_argument('--save-num-images', type=int, default=SAVE_NUM_IMAGES,
                        help='Number of images to save')
    parser.add_argument('--save-pred-every', type=int, default=SAVE_PRED_EVERY,
                        help='When to save summaries and checkpoints')
    parser.add_argument('--snapshot-dir', type=str, default=SNAPSHOT_DIR,
                        help='Where to save snapshots of the model')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Regularization parameter for L2-loss')
    parser.add_argument('--atrous-blocks', type=int, default=ATROUS_BLOCKS,
                        help='Number of continuous atrous blocks to link')
    return parser.parse_args()


def load_model(loader, sess, ckpt_path):
    checkpoint = tf.train.get_checkpoint_state(ckpt_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        print checkpoint.model_checkpoint_path
        loader.restore(sess, checkpoint.model_checkpoint_path)
        print("Restored model parameters from {}".format(ckpt_path))
    else:
        print 'Did not load model'


def save_model(saver, sess, logdir, global_step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=global_step)
    print('The checkpoint has been created for step {}'.format(sess.run(global_step)))


def main():
    # Create model and start training
    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    tf.set_random_seed(args.random_seed)

    # Create queue coordinator
    coord = tf.train.Coordinator()

    # Load Image Reader
    with tf.name_scope('create_inputs'):
        reader = ImageReader(
            args.index_loc,
            args.data_dir,
            args.mask_dir,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)

        image_batch, label_batch = reader.dequeue(args.batch_size)

    mode = tf.contrib.learn.ModeKeys.TRAIN
    net = DeepLabResNetModel(image_batch, mode, args.num_classes, args.atrous_blocks)

    raw_output = net.output

    # Trainable Variables
    restore_vars = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name]
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_labels(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False)
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    # Pixel-wise Softmax Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

    # Processed predictions: for visualization
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Image Summary
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    # preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)
    # labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)

    total_summary = tf.summary.image('images',
                                     tf.concat(axis=2, values=[images_summary]),
                                     max_outputs=args.save_num_images)
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())

    # Define loss and optimization parameters
    base_lr = tf.constant(args.learning_rate, tf.float64)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    increment_step = tf.assign(global_step, global_step + 1)
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - global_step / args.num_steps), args.power))

    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(increment_step, train_op_conv, train_op_fc_w, train_op_fc_b)

    # Set up session

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=3)
        if args.restore_from is not None:
            loader = tf.train.Saver()
            load_model(loader, sess, args.restore_from)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for step in range(args.num_steps):
            start_time = time.time()

            if step % args.save_pred_every == 0:
                feed = [reduced_loss, image_batch, label_batch, pred, total_summary, global_step, train_op]
                loss_value, images, labels, preds, summary, total_steps, _ = sess.run(feed)
                summary_writer.add_summary(summary, step)
                save_model(saver, sess, args.snapshot_dir, global_step)
            else:
                feed = [reduced_loss, global_step, train_op]
                loss_value, total_steps, _ = sess.run(feed)

            duration = time.time() - start_time
            print('global step: {:d}, step: {:d} \t loss = {:.3f}, ({:.3f} secs)'
                  .format(total_steps, step, loss_value, duration))
            print sess.run(learning_rate)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
