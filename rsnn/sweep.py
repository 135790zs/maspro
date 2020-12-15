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
    Categorical((False, True), name='update_input_weights'),
    # Categorical((False, True), name='traub_trick'),
    # Integer(1, 2, name='n_directions'),
    # Integer(1, 20, name="dt_refr"),
    # Integer(0, 5, name="delay"),
    # Real(0, 1, name="fraction_ALIF"),
    Real(0.01, 0.04, name='eta_init'),
    Real(0.7, 3, name='eta_slope'),
    Real(2, 10, name='eta_init_loss'),
    # Real(0.5, 2.5, name='thr'),
    Real(0.9, 0.99, name="alpha"),
    Real(0, 1, name="beta"),
    Real(0.1, 0.95, name="kappa"),
    Real(0.99, 0.9999, name="rho"),
    Real(0.1, 0.7, name="gamma"),
    Real(0, 1e-1, name="weight_decay"),
    Real(0, 1e-4, name="L2_reg"),
    Real(0, .1, name="FR_target"),
    Real(0, 300, name="FR_reg"),
    Real(0, .4, name="dropout"),
    # Integer(64, 600, name='N_R'),
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
                         n_calls=1000,
                         n_initial_points=20)
    # summarizing finding:
    print(result)
    print(f'Best Accuracy: {result.fun:.3f}')
    print(f'Best Parameters: {result.x}')
