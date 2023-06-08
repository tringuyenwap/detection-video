#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import tensorflow as tf
import pdb

from ae.cbam import *


def encoder(inputs_):
    with tf.variable_scope('encoder'):
        ### Encoder
        conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

        # Now 64x64x32
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 32x32x32
        conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 32x32x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 16x16x32
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 16x16x16
        encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
        return encoded, conv1, conv2, conv3


def encoder_shallow(inputs_):
    with tf.variable_scope('encoder'):
        ### Encoder
        conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

        # Now 64x64x32
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 32x32x32
        conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 32x32x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 16x16x32
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 16x16x8
        encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
        return encoded
def encoder_wider_latent(inputs_):
    with tf.variable_scope('encoder'):
        ### Encoder
        conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

        # Now 64x64x32
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 32x32x32
        conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 32x32x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 16x16x32
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 16x16x16
        encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
        return encoded
def encoder_wider(inputs_):
    with tf.variable_scope('encoder'):
        ### Encoder
        conv1 = tf.layers.conv2d(inputs=inputs_, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

        # Now 64x64x32
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 32x32x32
        conv2 = tf.layers.conv2d(inputs=maxpool1, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 32x32x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 16x16x32
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 16x16x16
        encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
        return encoded


def decoder(encoded, name, num_output_filter, conv1, conv2, conv3):
    with tf.variable_scope('decoder/' + name) as scope:
        # Now 8x8x16
        ### Decoder
        upsample1 = tf.image.resize_images(encoded, size=(encoded.shape[1] * 2, encoded.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 16x16x16
        conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)

        conv4 = conv4 + conv3
        # Now 16x16x16
        upsample2 = tf.image.resize_images(conv4, size=(conv4.shape[1] * 2, conv4.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 32x32x16
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        conv5 = conv5 + conv2
        # Now 32x32x32
        upsample3 = tf.image.resize_images(conv5, size=(conv5.shape[1] * 2, conv5.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 64x64x32
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        conv6 = conv6 + conv1
        # Now 64x64x32
        logits = tf.layers.conv2d(inputs=conv6, filters=num_output_filter, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        return logits


def decoder_wider(encoded, name, num_output_filter=1):
    with tf.variable_scope('decoder/' + name) as scope:
        # Now 8x8x16
        ### Decoder
        upsample1 = tf.image.resize_images(encoded, size=(encoded.shape[1] * 2, encoded.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 16x16x16
        conv4 = tf.layers.conv2d(inputs=upsample1, filters=64, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        # Now 16x16x16
        upsample2 = tf.image.resize_images(conv4, size=(conv4.shape[1] * 2, conv4.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 32x32x16
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=128, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        # Now 32x32x32
        upsample3 = tf.image.resize_images(conv5, size=(conv5.shape[1] * 2, conv5.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 64x64x32
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=256, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        # Now 64x64x32
        logits = tf.layers.conv2d(inputs=conv6, filters=num_output_filter, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        return logits


def decoder_mask(encoded, name):
    with tf.variable_scope('decoder/' + name) as scope:
        # Now 8x8x16
        ### Decoder
        upsample1 = tf.image.resize_images(encoded, size=(encoded.shape[1] * 2, encoded.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 16x16x16
        conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        # Now 16x16x16
        upsample2 = tf.image.resize_images(conv4, size=(conv4.shape[1] * 2, conv4.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 32x32x16
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        # Now 32x32x32
        upsample3 = tf.image.resize_images(conv5, size=(conv5.shape[1] * 2, conv5.shape[2] * 2),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 64x64x32
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        mask = create_mask(conv6)
        return mask


