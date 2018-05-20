"""Trains a simple convolutional net on the MNIST dataset.

Gets to 99.5% validation set accuracy.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import logging
import tefla
from tefla.core.learning import SupervisedLearner
from tefla.core.lr_policy import StepDecayPolicy
from tefla.core.mem_dataset import DataSet
from tefla.utils import util
from conv_model import model
from tefla.core import metrics

print('done importing')

import tensorflow as tf
np.random.seed(127)
tf.set_random_seed(127)

kappav2 = metrics.KappaV2(num_classes=5, batch_size=32)


def train():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

  width = 28
  height = 28

  train_images = mnist[0].images.reshape(-1, height, width, 1)
  train_labels = mnist[0].labels

  validation_images = mnist[1].images.reshape(-1, height, width, 1)
  validation_labels = mnist[1].labels

  data_set = DataSet(train_images, train_labels, validation_images, validation_labels)

  training_cnf = {
      'classification':
      True,
      'validation_scores': [('validation accuracy', tf.contrib.metrics.accuracy),
                            ('validation kappa', kappav2.metric)],
      'num_epochs':
      50,
      'batch_size_train':
      32,
      'batch_size_test':
      32,
      'input_size': (28, 28, 1),
      'lr_policy':
      StepDecayPolicy(schedule={
          0: 0.01,
          30: 0.001,
      })
  }

  learner = SupervisedLearner(
      model,
      training_cnf,
      classification=training_cnf['classification'],
      is_summary=True,
      num_classes=10)
  learner.fit(data_set, weights_from=None, start_epoch=1, summary_every=10)


if __name__ == '__main__':
  train()
