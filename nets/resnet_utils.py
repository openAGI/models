"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tefla.core.layers import dilated_conv2d, conv2d, fully_connected, max_pool, global_avg_pool, batch_norm_tf as batch_norm
from tefla.utils import util


class Block(collections.namedtuple('Block', ['name', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    name: The name/scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """


def subsample(inputs, factor, name=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    name: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return max_pool(inputs, filter_size=(1, 1), stride=(factor, factor), name=name)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, name=None, **kwargs):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    name: name.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return dilated_conv2d(
        inputs,
        num_outputs,
        filter_size=(kernel_size, kernel_size),
        dilation=rate,
        padding='SAME',
        name=name,
        **kwargs)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return dilated_conv2d(
        inputs,
        num_outputs,
        filter_size=kernel_size,
        stride=stride,
        dilation=rate,
        padding='VALID',
        name=name,
        **kwargs)


def stack_blocks_dense(net, blocks, output_stride=None, **kwargs):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the ResNet
  output_stride, which is the ratio of the input to output spatial resolution.
  This is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Most ResNets consist of 4 ResNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive ResNet blocks. This results
  to a nominal ResNet output_stride equal to 8. If we set the output_stride to
  half the nominal network stride (e.g., output_stride=4), then we compute
  responses twice.

  Control of the output feature density is implemented by atrous convolution.

  Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
          element is a ResNet `Block` object describing the units in the `Block`.
      output_stride: If `None`, then the output will be computed at the nominal
          network stride. If output_stride is not `None`, it specifies the requested
          ratio of input to output spatial resolution, which needs to be equal to
          the product of unit strides from the start up to some level of the ResNet.
          For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
          then valid values for the output_stride are 1, 2, 6, 24 or None (which
          is equivalent to output_stride=24).
      outputs_collections: Collection to add the ResNet block outputs.

  Returns:
      net: Output tensor with stride equal to the specified output_stride.

  Raises:
      ValueError: If the target output_stride is not valid.
  """
  current_stride = 1
  # The dilated convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.name, 'block', [net]):
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride = unit

          # If we have reached the target output_stride, then we need to employ
          # dilated convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, unit_depth, unit_depth_bottleneck, 1, rate=rate, **kwargs)
            rate *= unit_stride

          else:
            net = block.unit_fn(
                net, unit_depth, unit_depth_bottleneck, unit_stride, rate=1, **kwargs)
            current_stride *= unit_stride

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


def bottleneck_se(inputs, depth, depth_bottleneck, stride, rate=1, name=None, **kwargs):
  """SE Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    name: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  is_training = kwargs.get('is_training')
  reuse = kwargs.get('reuse')
  with tf.variable_scope(name, 'bottleneck_se', [inputs]):
    depth_in = util.last_dimension(inputs.get_shape(), min_rank=4)
    preact = batch_norm(
        inputs, activation_fn=tf.nn.relu, name='preact', is_training=is_training, reuse=reuse)
    if depth * 4 == depth_in:
      shortcut = subsample(preact, stride, 'shortcut')
    else:
      shortcut = conv2d(
          preact,
          depth * 4,
          is_training,
          reuse,
          filter_size=(1, 1),
          stride=(stride, stride),
          batch_norm=None,
          activation=None,
          name='shortcut')

    residual = conv2d(preact, depth, filter_size=(1, 1), stride=(1, 1), name='conv1', **kwargs)
    residual = conv2d(
        residual, depth, filter_size=(3, 3), stride=(stride, stride), name='conv2', **kwargs)
    residual = conv2d(
        residual,
        depth * 4,
        is_training,
        reuse,
        filter_size=(1, 1),
        stride=(1, 1),
        batch_norm=None,
        activation=None,
        name='conv3')

    squeeze = global_avg_pool(residual, name='se_global_avg_pool')
    squeeze = fully_connected(squeeze, depth // 4, is_training, reuse, name='fc1')
    squeeze = fully_connected(squeeze, depth * 4, is_training, reuse, name='fc2')
    squeeze = tf.nn.sigmoid(squeeze, name='se_fc_sigmoid')
    residual = residual * tf.reshape(squeeze, [-1, 1, 1, depth * 4])
    return residual + shortcut
