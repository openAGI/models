'''hyperparameter tuning with hyperband'''
import pickle
import click
from tefla.core.mem_dataset import DataSet
from tefla.core.iter_ops import create_training_iters
from tefla.core.learning import SupervisedLearner
from tefla.da.standardizer import NoOpStandardizer
from tefla.utils import util
from tefla.core.hyperband import Hyperband
from tensorflow.examples.tutorials.mnist import input_data


# pylint: disable=no-value-for-parameter
@click.command()
@click.option('--model',
              default='model.py', show_default='True', help='Relative path to model.')
@click.option('--training_cnf',
              default='training_cnf.py', show_default=True, help='Relative path to training config file.')
@click.option('--tuning_cnf',
              default='tuning_cnf.py', show_default=True, help='Relative path to training config file.')
@click.option('--results_dir',
              default='./results.pkl', show_default=True, help='Relative path to hyperband results directory.')
@click.option('--verbose',
              default=3, show_default=True, help='Verbose level.')
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def main(model, training_cnf, tuning_cnf, results_dir, verbose):
  """main function to call hyperband
  """
  # keeping the CL arguments in a dict for passing to multiple functions
  args = {
      'model': model,
      'training_cnf': training_cnf,
      'tuning_cnf': tuning_cnf,
      'verbose': verbose,
  }
  hyperband = Hyperband(try_config, args)
  results = hyperband.run()
  with open(results_dir, "wb") as fil:
    pickle.dump(results, fil)
  print('Hyperband restults are saved to {}'.format(
      results_dir))


def try_config(args, cnf):
  """For trying out configurations.

  Args:
      args: command line arguments regarding training
      cnf: training configuration sampled from hyperband search space

  Returns:
      a dictionary containing final loss value and early stop flag
  """
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

  width = 28
  height = 28

  train_images = mnist[0].images
  train_labels = mnist[0].labels

  validation_images = mnist[1].images
  validation_labels = mnist[1].labels

  data_set = DataSet(train_images, train_labels, validation_images, validation_labels)

  model_def = util.load_module(args['model'])
  model = model_def.model

  learner = SupervisedLearner(
      model,
      cnf,
      classification=cnf['classification'],
      is_summary=False,
      num_classes=10,
      verbosity=args['verbose'],
      is_early_stop=cnf.get('is_early_stop', True))
  _early_stop, _loss = learner.fit(data_set, weights_from=None, start_epoch=1)

  return {'early_stop': _early_stop, 'loss': _loss}


if __name__ == "__main__":
  main()
