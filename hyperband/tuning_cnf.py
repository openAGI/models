from hyperopt import hp

max_conv_layers = 0
max_fc_layers = 3

space = { 
          'max_conv_layers': max_conv_layers,
          'max_fc_layers': max_fc_layers,
          'batch_size_train': hp.choice( 'bs', (16,32,64,128,256)),
          'batch_size_test': hp.choice( 'bs', (16,32)),
          'n_conv_layers': hp.quniform( 'c', 1, max_conv_layers, 1 ), 
          'n_fc_layers': hp.quniform( 'l', 1, max_fc_layers, 1 ), 
          'init': hp.choice( 'i', ( 'uniform', 'normal', 'glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal' )),
          'optimizer': hp.choice( 'o', ( 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax' )),
          'batch_norm': hp.choice( 'b', ( 'True', 'False' ))
}

# Layerwise config
if max_conv_layers>0:
  for i in range( 1, max_conv_layers + 1 ):
    space[ 'conv_layer_{}_size'.format( i )] = hp.quniform( 'ls{}'.format( i ), 2, 200, 2 )
    space[ 'conv_layer_{}_activation'.format( i )] = hp.choice( 'a{}'.format( i ), ( 'relu', 'sigmoid', 'tanh' ))
    space[ 'conv_layer_{}_maxpool'.format( i )] = hp.choice( 'm{}'.format( i ), ( 'True', 'False')) 

if max_fc_layers>0:
  for i in range( 1, max_fc_layers + 1 ):
    space[ 'fc_layer_{}_size'.format( i )] = hp.quniform( 'ls{}'.format( i ), 2, 200, 2 )
    space[ 'fc_layer_{}_activation'.format( i )] = hp.choice( 'a{}'.format( i ), ( 'relu', 'sigmoid', 'tanh' ))
    space[ 'fc_layer_{}_dropout'.format( i )] = hp.quniform( 'd{}'.format( i ), 0.1, 0.5, 0.1 ) 