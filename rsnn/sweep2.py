# example of bayesian optimization with scikit-optimize
from numpy import mean
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from rsnn import main
from config import cfg
import datetime
import csv

file_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
fname = f'../sweep_results-{file_id}.csv'

def rsnn_aux(**params):
    cfg0 = dict(cfg)
    for k, v in params.items():
        cfg0[k] = v
    res = main(cfg0)
    return res

# define the space of hyperparameters to search
search_space = [Integer(64, 128, name='N_R'),
                Integer(1, 3, name='Repeats'),
                Categorical((False, True), name='traub_trick'),
                Integer(1, 2, name='n_directions')]

with open(fname, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow([x.name for x in search_space] + ['V-Error'])

# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):

    res = rsnn_aux(**params)
    with open(fname, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(list(params.values()) + [res])

    return res

# perform optimization
result = gp_minimize(evaluate_model,
                     search_space,
                     n_calls=100,
                     n_initial_points=20)
# summarizing finding:
print(result)
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print(f'Best Parameters: {result.x}')
