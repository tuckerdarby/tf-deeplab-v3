import tensorflow as tf
import tensorflow.contrib.layers as layers

from net_base import BaseModel


class DeepLabResNetModel(BaseModel):
    def __init__(self, inbound, mode, num_classes, atrous_blocks=22):
        super(DeepLabResNetModel, self).__init__(mode)
        self.num_classes = num_classes
        self.inbound = inbound

        # 1 - Inbound
        # self.inbound = tf.placeholder(tf.float32, (321, 321, 3))
        with tf.variable_scope('1a'):
            conv = layers.conv2d(self.inbound, 64, 7, 2, activation_fn=None)
            bn = layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=self.training)
            pool = layers.max_pool2d(bn, 3, 2)
        block_end = pool

        # 2
        ## 2a - Branch 1
        r2a_b1 = self._a_branch(block_end, 256, 1, 1, scope='2ab1')
        ## 2a - Branch 2
        r2a_block = self._conv_block(block_end, (64, 64, 256), (1, 3, 1), (1, 1, 1), scope='2a')
        r2a_relu1 = self._add_block(r2a_b1, r2a_block, scope='2a')
        ## 2b - Branch 2
        r2b_block = self._conv_block(r2a_relu1, (64, 64, 256), (1, 3, 1), (1, 1, 1), scope='2b')
        r2b_relu1 = self._add_block(r2a_relu1, r2b_block, scope='2b')
        ## 2c - Branch 2
        r2c_block = self._conv_block(r2b_relu1, (64, 64, 256), (1, 3, 1), (1, 1, 1), scope='2c')
        r2c_relu1 = self._add_block(r2b_relu1, r2c_block, scope='2c')
        block_end = r2c_relu1

        # 3
        ## 3a - Branch 1
        r3a_b1 = self._a_branch(block_end, 512, 1, 2, scope='3ab1')
        ## 3a - Branch 2
        r3a_block = self._conv_block(r2c_relu1, (128, 128, 512), (1, 3, 1), (2, 1, 1), scope='3a')
        r3a_relu = self._add_block(r3a_b1, r3a_block, scope='3a')
        ## 3b1 - Branch 2
        r3b1_block = self._conv_block(r3a_relu, (128, 128, 512), (1, 3, 1), (1, 1, 1), scope='3b1')
        r3b1_relu = self._add_block(r3a_relu, r3b1_block, scope='3b1')
        ## 3b2 - Branch 2
        r3b2_block = self._conv_block(r3b1_relu, (128, 128, 512), (1, 3, 1), (1, 1, 1), scope='3b2')
        r3b2_relu = self._add_block(r3b1_relu, r3b2_block, scope='3b2')
        ## 3b3 - Branch 2
        r3b3_block = self._conv_block(r3b2_relu, (128, 128, 512), (1, 3, 1), (1, 1, 1), scope='3b3')
        r3b3_relu = self._add_block(r3b2_relu, r3b3_block, scope='3b3')
        block_end = r3b3_relu

        # 4
        ## 4a - Branch 1
        r4a_b1 = self._a_branch(block_end, 1024, 1, 1, scope='4ab1')
        ## 4a - Branch 2
        r4a_block = self._aconv_block(r4a_b1, (256, 256, 1024), (1, 3, 1), (1, 2, 1), scope='4a')
        r4a_relu = self._add_block(r4a_b1, r4a_block, scope='4a')
        ## 4b - Branch 2
        r4b_relu = r4a_relu
        for i in range(1, atrous_blocks + 1):
            scope = '4b' + str(i)
            r4b_block = self._aconv_block(r4b_relu, (256, 256, 1024), (1, 3, 1), (1, 2, 1), scope=scope)
            r4b_relu = self._add_block(r4b_relu, r4b_block, scope=scope)
        block_end = r4b_relu

        # 5
        ## 5a - Branch 1
        r5a_b1 = self._a_branch(block_end, 2048, 1, 1, scope='5ab1')
        ## 5a - Branch 2
        r5a_block = self._aconv_block(r5a_b1, (512, 512, 2048), (1, 3, 1), (1, 4, 1), scope='5a')
        r5a_relu = self._add_block(r5a_b1, r5a_block, scope='5a')
        ## 5b - Branch 2
        r5b_block = self._aconv_block(r5a_relu, (512, 512, 2048), (1, 3, 1), (1, 4, 1), scope='5b')
        r5b_relu = self._add_block(r5a_relu, r5b_block, scope='5b')
        ## 5c - Branch 2
        r5c_block = self._aconv_block(r5b_relu, (512, 512, 2048), (1, 3, 1), (1, 4, 1), scope='5c')
        r5c_relu = self._add_block(r5b_relu, r5c_block, scope='5c')
        block_end = r5c_relu

        # 5 - Atrous 'FCs'
        with tf.variable_scope('fc'):
            fc_c0 = self._aconv_layer(block_end, num_classes, 3, 6)
            fc_c1 = self._aconv_layer(block_end, num_classes, 3, 12)
            fc_c2 = self._aconv_layer(block_end, num_classes, 3, 18)
            fc_c3 = self._aconv_layer(block_end, num_classes, 3, 24)
            fc = tf.add(fc_c0, fc_c1)
            fc = tf.add(fc, fc_c2)
            fc = tf.add(fc, fc_c3)

        self.output = fc
