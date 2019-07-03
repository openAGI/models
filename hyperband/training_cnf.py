import tensorflow as tf
from tefla.core.lr_policy import StepDecayPolicy

cnf = {
      'classification':
      True,
      'validation_scores': [('accuracy', tf.metrics.accuracy),
                            ('kappa', tf.contrib.metrics.cohen_kappa)],
      'num_epochs':
      50,
      'batch_size_train':
      32,
      'batch_size_test':
      32,
      'input_size': (28 * 28, ),
      'lr_policy':
      StepDecayPolicy(schedule={
          0: 0.01,
          30: 0.001,
      })
  }