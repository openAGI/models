"""Contains definitions for the SE Networks.

the main pathway. Also see [2; Fig. 4e].
Typical use:

SEResNet-101 for image classification into 5 classes:

    # inputs has shape [batch, 224, 224, 3]
    net, end_points = senet.seresnet_v2_101(inputs, is_training, reuse, 5)

SEResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
      net, end_points = senet.resnet_v1_101(inputs, 21, is_training=False, global_pool=False, output_stride=16)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.core import initializers as initz
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import conv2d, max_pool, global_avg_pool, prelu, softmax, batch_norm_tf as batch_norm
from tefla.utils import util

from models import resnet_utils

image_size = (128, 128)
crop_size = (112, 112)


def seresnet_v2(inputs, is_training, reuse,
                blocks,
                num_classes=None,
                global_pool=True,
                output_stride=None,
                include_root_block=True,
                name=None):
    """Generator for v2 (preactivation) ResNet models.

    This function generates a family of ResNet v2 models. See the resnet_v2_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.

    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.

    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      is_training: whether is training or not.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      name: Optional variable_scope.


    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    logits_args = make_args(
        activation=None, w_init=initz.he_normal(scale=1), **common_args)
    pred_args = make_args(
        activation=None, **common_args)
    pool_args = make_args(padding='SAME', **common_args)

    with tf.variable_scope(name, 'resnet_v2', [inputs], reuse=reuse):
        net = inputs
        if include_root_block:
            if output_stride is not None:
                if output_stride % 4 != 0:
                    raise ValueError(
                        'The output_stride needs to be a multiple of 4.')
                output_stride /= 4
            # We do not include batch normalization or activation functions in
            # conv1 because the first ResNet unit will perform these. Cf.
            # Appendix of [2].
            net = resnet_utils.conv2d_same(
                net, 64, 7, 1, rate=2, name='conv1', **common_args)
            net = max_pool(net, name='pool1', **pool_args)
        net = resnet_utils.stack_blocks_dense(
            net, blocks, output_stride, **conv_args)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = batch_norm(net, name='postnorm',
                         is_training=is_training, reuse=reuse)
        net = tf.nn.relu(net, name='postnorm_activation')
        if num_classes is not None:
            net = conv2d(net, num_classes, filter_size=(
                1, 1), name='num_clasess_conv2d', **logits_args)
        if global_pool:
            # Global average pooling.
            net = global_avg_pool(net, name='logits', **common_args)
        if num_classes is not None:
            predictions = softmax(net, name='predictions', **common_args)

        return end_points(is_training)


def model(inputs, is_training, reuse,
          num_classes=None,
          global_pool=True,
          output_stride=None,
          name='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_utils.Block(
            'block1', resnet_utils.bottleneck_se, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', resnet_utils.bottleneck_se, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', resnet_utils.bottleneck_se, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', resnet_utils.bottleneck_se, [(2048, 512, 1)] * 3)]
    return seresnet_v2(inputs, is_training, reuse, blocks, num_classes,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, name=name)


def seresnet_v2_101(inputs, is_training, reuse,
                    num_classes=None,
                    global_pool=True,
                    output_stride=None,
                    name='resnet_v2_101'):
    """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_utils.Block(
            'block1', resnet_utils.bottleneck_se, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', resnet_utils.bottleneck_se, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', resnet_utils.bottleneck_se, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', resnet_utils.bottleneck_se, [(2048, 512, 1)] * 3)]
    return seresnet_v2(inputs, is_training, reuse, blocks, num_classes,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, name=name)


def seresnet_v2_152(inputs, is_training, reuse,
                    num_classes=None,
                    global_pool=True,
                    output_stride=None,
                    name='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_utils.Block(
            'block1', resnet_utils.bottleneck_se, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', resnet_utils.bottleneck_se, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', resnet_utils.bottleneck_se, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', resnet_utils.bottleneck_se, [(2048, 512, 1)] * 3)]
    return seresnet_v2(inputs, is_training, reuse, blocks, num_classes,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, name=name)


def seresnet_v2_200(inputs, is_training, reuse,
                    num_classes=None,
                    global_pool=True,
                    output_stride=None,
                    name='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_utils.Block(
            'block1', resnet_utils.bottleneck_se, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', resnet_utils.bottleneck_se, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', resnet_utils.bottleneck_se, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', resnet_utils.bottleneck_se, [(2048, 512, 1)] * 3)]
    return seresnet_v2(inputs, is_training, reuse, blocks, num_classes,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, name=name)
