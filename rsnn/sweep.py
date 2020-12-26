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
    # Categorical((False, True), name='update_input_weights'),
    # Categorical((False, True), name='traub_trick'),
    # Integer(1, 2, name='n_directions'),
    # Integer(2, 50, name="dt_refr"),
    # Integer(0, 5, name="delay"),
    Real(0.2, 1, name="fraction_ALIF"),
    Real(1e-5, 5e-2, name='eta_b_out'),
    Real(1e-4, 1e-1, name='eta_init'),
    # Real(0.5, 3, name='eta_slope'),
    # Real(0, 10, name='eta_init_loss'),
    Real(0.8, 2.5, name='thr'),
    # Real(0.9, 0.99, name="alpha"),
    # Real(0.05, 2.5, name="beta"),
    # Real(0.5, 0.9, name="kappa"),
    # Real(0.95, 1, name="rho"),
    # Real(0.1, 0.7, name="gamma"),
    # Real(0, 1e-1, name="weight_decay"),
    # Real(0, 1e-4, name="L2_reg"),
    # Real(0, .2, name="FR_target"),
    Real(0, 50, name="FR_reg"),
    # Real(.5, .95, name="dropout"),
    # Real(0.5, 1.5, name="weight_scaling"),
    # Real(0.3, 3, name="softmax_factor"),
    # Real(1e-9, 1e-4, name="adam_eps"),
    # Integer(64, 600, name='N_R'),
]

file_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
fname = f'../sweeps/sweep_results-{file_id}.csv'


def rsnn_aux(**params):
    cfg0 = dict(cfg)
    for k, v in params.items():
        cfg0[k] = v
    optVerr, final_p_wrong = main(cfg0)
    return optVerr, final_p_wrong


@use_named_args(search_space)
def evaluate_model(**params):

    optVerr, final_p_wrong = rsnn_aux(**params)
    with open(fname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile,
                               delimiter=',',
                               quotechar='"',
                               quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(list(params.values()) + [optVerr] + [final_p_wrong])

    # Return minimization
    return final_p_wrong


if __name__ == "__main__":
    with open(fname, 'w', newline='') as new_csv:
        varname_writer = csv.writer(new_csv,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        varname_writer.writerow([x.name for x in search_space] + ['V-Error'] + ['T-Wrongs'])

    # perform optimization
    result = gp_minimize(evaluate_model,
                         search_space,
                         n_calls=1000,
                         n_initial_points=50)
    # summarizing finding:
    print(result)
    print(f'Best Accuracy: {result.fun:.3f}')
    print(f'Best Parameters: {result.x}')
