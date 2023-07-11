
from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def conv2d(inputs, filters, kernel_size, strides=1):
    padding = 'same' if strides == 1 else 'valid'
    return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(inputs)

def darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)
        net = layers.Add()([net, shortcut])
        return net

    net = conv2d(inputs, 32, 3, strides=1)
    net = conv2d(net, 64, 3, strides=2)
    net = res_block(net, 32)
    net = conv2d(net, 128, 3, strides=2)

    for _ in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    for _ in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    for _ in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    for _ in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3

def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net

def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    return tf.image.resize(inputs, size=(new_height, new_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
