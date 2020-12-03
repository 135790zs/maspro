# example of bayesian optimization with scikit-optimize
import datetime
import csv
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from rsnn import main
from config import cfg

# define the space of hyperparameters to search
search_space = [
    # Categorical(("random", "symmetric", "adaptive"), name='eprop_type'),
    # Categorical(("Adam", "SGD"), name='optimizer'),
    # Categorical((False, True), name='traub_trick'),
    # Real(0, 1, name="fraction_ALIF"),
    # Real(10, 500, name="theta_adaptation"),
    # Integer(1, 2, name='n_directions'),
    # Integer(0, 5, name="delay"),
    # Real(2, 30, name="theta_membrane"),
    # Real(0.5, 6, name="theta_output"),
    # Real(0.05, 0.5, name="beta"),
    Real(0.3, 1, name="gamma"),
    Real(0.05, 0.5, name="weight_decay"),
    Real(0, 1e-4, name="L2_reg"),
    # Real(0, 100, name="FR_reg"),
    Real(0.05, .2, name="FR_target"),
    Real(0, .9, name="dropout"),
    # Integer(128, 220, name='N_R'),
]

file_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
fname = f'../sweeps/sweep_results-{file_id}.csv'


def rsnn_aux(**params):
    cfg0 = dict(cfg)
    for k, v in params.items():
        cfg0[k] = v
    res = main(cfg0)
    return res


@use_named_args(search_space)
def evaluate_model(**params):

    res = rsnn_aux(**params)
    with open(fname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile,
                               delimiter=',',
                               quotechar='"',
                               quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(list(params.values()) + [res])

    return res


if __name__ == "__main__":
    with open(fname, 'w', newline='') as new_csv:
        varname_writer = csv.writer(new_csv,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        varname_writer.writerow([x.name for x in search_space] + ['V-Error'])

    # perform optimization
    result = gp_minimize(evaluate_model,
                         search_space,
                         n_calls=500,
                         n_initial_points=20)
    # summarizing finding:
    print(result)
    print(f'Best Accuracy: {result.fun:.3f}')
    print(f'Best Parameters: {result.x}')
