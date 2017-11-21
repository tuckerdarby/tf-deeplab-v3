import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn import ModeKeys


class BaseModel(object):
    def __init__(self, mode):
        self.training = ModeKeys.TRAIN == mode
        self.weights = []
        self.biases = []

    def _create_weight(self, name, shape):
        init = layers.xavier_initializer_conv2d(dtype=tf.float32)
        return tf.Variable(init(shape=shape), name=name)

    def _create_bias(self, name, shape):
        init = tf.constant_initializer(value=0.0, dtype=tf.float32)
        return tf.Variable(init(shape=shape), name=name)

    def _a_branch(self, inbound, output_size, kernel_size, stride, scope):
        conv = layers.conv2d(inbound, output_size, kernel_size, stride, activation_fn=None)
        bn = layers.batch_norm(conv, activation_fn=None, is_training=self.training)
        return bn

    def _aconv_layer(self, inbound, output_size, kernel_size, rate):
        aconv_shape = [kernel_size, kernel_size, inbound.get_shape()[3].value, output_size]
        weight = self._create_weight('weights', aconv_shape)
        bias = self._create_bias('biases', [output_size])
        atrous = tf.nn.atrous_conv2d(inbound, weight, rate, padding='SAME')
        atrous = tf.nn.bias_add(atrous, bias)
        self.weights.append(weight)
        self.biases.append(bias)
        return atrous

    def _aconv_block(self, inbound, output_sizes, kernel_sizes, strides, scope):
        with tf.variable_scope(scope):
            conv = layers.conv2d(inbound, output_sizes[0], kernel_sizes[0], strides[0], activation_fn=None)
            bn = layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=self.training)
            atrous = self._aconv_layer(bn, output_sizes[1], kernel_sizes[1], strides[1])
            bn = layers.batch_norm(atrous, activation_fn=tf.nn.relu, is_training=self.training)
            conv = layers.conv2d(bn, output_sizes[2], kernel_sizes[2], strides[2], activation_fn=None)
            bn = layers.batch_norm(conv, activation_fn=None, is_training=self.training)
        return bn

    def _conv_block(self, inbound, output_sizes, kernel_sizes, strides, scope):
        with tf.variable_scope(scope):
            conv = layers.conv2d(inbound, output_sizes[0], kernel_sizes[0], strides[0], activation_fn=None)
            bn = layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=self.training)
            conv = layers.conv2d(bn, output_sizes[1], kernel_sizes[1], strides[1], activation_fn=None)
            bn = layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=self.training)
            conv = layers.conv2d(bn, output_sizes[2], kernel_sizes[2], strides[2], activation_fn=None)
            bn = layers.batch_norm(conv, activation_fn=None, is_training=self.training)
        return bn

    def _add_block(self, first, second, scope):
        with tf.variable_scope(scope):
            add = tf.add(first, second)
            relu = tf.nn.relu(add, name='relu')
        return relu