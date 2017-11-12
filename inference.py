import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
import numpy as np

from net_help import load_model
from net import DeepLabResNetModel
from utils import decode_labels, prepare_labels
from image_reader import ImageReader
from defaults import *


def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab Inference")
    parser.add_argument('img_path',
                        help="Path to image file")
    parser.add_argument("--model-weights", type=str, default=RESTORE_FROM,
                        help="Path to the file with model weights")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to saved inferenced image")
    return parser.parse_args()


def main():
    args = get_arguments()
    # Read Image
    img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # Convert RGB to BGR
    red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[blue, green, red]), dtype=tf.float32)
    # Extract mean
    img -= IMG_MEAN

    # Create Network
    net = DeepLabResNetModel(tf.expand_dims(img, dim=0), ModeKeys.TRAIN, args.num_classes)

    # Predictions
    raw_output = net.output
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Init
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        restorer = tf.train.Saver()
        load_model(restorer, sess, args.model_weights)
        preds = sess.run(pred)
        msk = decode_labels(preds, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        im.save(args.save_dir + 'mask.png')

        print 'Image saved'


if __name__ == '__main__':
    main()