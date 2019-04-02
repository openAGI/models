import numpy as np
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import conv2d, fully_connected, max_pool, softmax, prelu, dropout

width = 32
height = 32

image_size = (32, 32)
crop_size = (28, 28)

def layer_config(config, layer, layer_type='conv'):
  layer = layer_type + '_' + 'layer_'+ str(layer)
  if layer_type == 'conv':
    return config[layer+'_activation'], config[layer+'_size'], config[layer+'_maxpool']
  else:
    return config[layer+'_activation'], config[layer+'_size'], config[layer+'_dropout']

def model(x, is_training, reuse, num_classes=10, **config):
  common_args = common_layer_args(is_training, reuse)
  logit_args = make_args(activation=None, **common_args)

  if config['max_conv_layers']>0:
    for i in range(1, config['n_conv_layers']+1):
      activation, size, maxpool = layer_config(config, i, layer_type='conv')
      conv_args = make_args(batch_norm=bool(config['batch_norm']), activation=prelu, **common_args)
      x = conv2d(x, size, name='conv{}'.format(i), **conv_args)
      if maxpool:
        x = max_pool(x, name='pool{}'.format(i), **common_args)

  if config['max_fc_layers']>0:
    for i in range(1, config['n_fc_layers']+1):
      activation, size, _dropout = layer_config(config, i, layer_type='fc')
      fc_args = make_args(activation=prelu, **common_args)
      x = fully_connected(x, n_output=size, name='fc{}'.format(i), **fc_args)
      x = dropout(x, drop_p=np.round(_dropout, 2), name='dropout{}'.format(i), **common_args)

  logits = fully_connected(x, n_output=num_classes, name="logits", **logit_args)
  predictions = softmax(logits, name='predictions', **common_args)
  return end_points(is_training)