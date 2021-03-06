from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import conv2d, separable_conv2d, fully_connected, max_pool, prelu, repeat
from tefla.core.layers import input, softmax, dropout, global_avg_pool, avg_pool_2d, batch_norm_tf as batch_norm

image_size = (128, 128)
crop_size = (112, 112)


def bottleneck_v1(inputs, num_unit=128, name=None, **kwargs):
  is_training = kwargs.get('is_training')
  reuse = kwargs.get('reuse')
  with tf.variable_scope(name, 'bottleneck_v2', [inputs]):
    residual = conv2d(
        inputs, num_unit, filter_size=(1, 1), stride=(2, 2), name='conv_res', **kwargs)
    net = tf.nn.relu(inputs)
    net = separable_conv2d(
        net, num_unit, filter_size=(3, 3), stride=(1, 1), name='sconv1', **kwargs)
    net = separable_conv2d(
        net,
        num_unit,
        is_training,
        reuse,
        filter_size=(3, 3),
        stride=(1, 1),
        batch_norm=True,
        activation=None,
        name='sconv2')

    net = max_pool(net, name='maxpool')
    output = net + residual
    return output


def bottleneck_v2(inputs, num_unit=128, name=None, **kwargs):
  is_training = kwargs.get('is_training')
  reuse = kwargs.get('reuse')
  with tf.variable_scope(name, 'bottleneck_v2', [inputs]):
    net = tf.nn.relu(inputs)
    net = separable_conv2d(
        net, num_unit, filter_size=(3, 3), stride=(1, 1), name='sconv1', **kwargs)
    net = separable_conv2d(
        net, num_unit, filter_size=(3, 3), stride=(1, 1), name='sconv2', **kwargs)
    net = separable_conv2d(
        net,
        num_unit,
        is_training,
        reuse,
        filter_size=(3, 3),
        stride=(1, 1),
        batch_norm=True,
        activation=None,
        name='sconv3')

    # net = max_pool(net, name='maxpool')
    output = net + inputs
    return output


def model(inputs, is_training, reuse, dropout_keep_prob=1, num_classes=5, **kwargs):
  common_args = common_layer_args(is_training, reuse)
  conv_args = make_args(untie_biases=True, batch_norm=batch_norm, **common_args)
  logit_args = make_args(activation=prelu, **common_args)

  net = conv2d(inputs, 32, filter_size=(3, 3), stride=(2, 2), name='conv1', **conv_args)
  net = conv2d(net, 64, name='conv2', **conv_args)
  net = bottleneck_v1(net, num_unit=128, name='block_v1_1', **conv_args)
  net = bottleneck_v1(net, num_unit=256, name='block_v1_2', **conv_args)
  net = bottleneck_v1(net, num_unit=728, name='block_v1_3', **conv_args)

  for i in range(8):
    prefix = 'block_v2_' + str(i + 5)
    net = bottleneck_v2(net, num_unit=728, name=prefix, **conv_args)

  net = bottleneck_v1(net, num_unit=1024, name='block_v1_4', **conv_args)
  net = separable_conv2d(net, 1536, filter_size=(3, 3), stride=(1, 1), name='sconv1', **conv_args)
  net = separable_conv2d(net, 2048, filter_size=(3, 3), stride=(1, 1), name='sconv2', **conv_args)
  with tf.variable_scope('Logits'):
    net = global_avg_pool(net, name='AvgPool_1a')
    net = dropout(net, is_training, drop_p=1 - dropout_keep_prob, name='Dropout_1b')
    logits = fully_connected(net, num_classes, name='logits', **logit_args)
    predictions = softmax(logits, name='predictions', **common_args)
  return end_points(is_training)
