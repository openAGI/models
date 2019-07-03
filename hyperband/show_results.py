import click
import pickle, json
import numpy as np


@click.command()
@click.option(
    '--results_dir',
    default='./results.pkl',
    show_default=True,
    help='Path to hyperband results directory.')
@click.option('--n_results',
              default=3, show_default=True, help='No. of results_to_show.')
def main(results_dir, n_results):
  with open('./results.pkl', 'rb') as handle:
    results = pickle.load(handle)
  # print(results)

  for r in sorted( results, key = lambda x: x['loss'] )[:n_results]:
    print ("loss: {} | {} seconds | {:.1f} iterations | run {} ".format( 
        np.round(r['loss'], 4), r['seconds'], r['iterations'], r['counter'] ))
    print( json.dumps(r['params'], indent=4, sort_keys=False) )
if __name__ == '__main__':
  main()