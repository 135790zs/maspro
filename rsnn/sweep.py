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

    # Categorical(('random', 'symmetric', 'adaptive'), name='eprop_type'),
    # Categorical((False, True), name='v_fix'),
    # Categorical((False, True), name='v_fix_psi'),
    # Categorical((False, True), name='uniform_dist'),
    Integer(2, 4, name="dt_refr"),
    Real(0.2, 1, name="fraction_ALIF"),
    # Real(0.01, 0.1, name="IzhV1"),
    # Real(2, 8, name="IzhV2"),
    # Real(30, 500, name="IzhV3"),
    # Real(0.001, 0.01, name="IzhA1"),
    # Real(-0.05, -0.005, name="IzhA2"),
    # Real(-90, -30, name="IzhReset"),
    # Real(10, 50, name="thr"),
    Real(1, 3, name='thr'),
    Real(0.5, 0.99, name="alpha"),
    Real(0, 2.5, name="beta"),
    Real(0.1, 0.9, name="kappa"),
    Real(0.9, 0.9999, name="rho"),
    Real(0.1, 1, name="gamma"),
    # Real(0, 0.05, name="weight_decay"),
    # Real(0, 1e-4, name="L2_reg"),
    Real(0, .1, name="FR_target"),
    Real(0, 200, name="FR_reg"),
    # Real(0.001, 0.03, name="eta_W_in"),
    # Real(0.001, 0.03, name="eta_W_rec"),
    # Real(0.001, 0.03, name="eta_out"),
    # Real(0.001, 0.03, name="eta_bias"),
    # Real(0.1, 1, name="weight_scaling"),
    Real(1e-9, 1e-4, name="adam_eps"),
    Real(0.8, 0.999, name="adam_beta1"),
    Real(0.8, 0.9999, name="adam_beta2"),
    # Real(0.2, 1, name="weight_scaling"),
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
        varname_writer.writerow([x.name for x in search_space] + ['T-Error'] + ['T-Wrongs'])

    # perform optimization
    result = gp_minimize(evaluate_model,
                         search_space,
                         n_calls=1000,
                         n_initial_points=30)
    # summarizing finding:
    print(result)
    print(f'Best Accuracy: {result.fun:.3f}')
    print(f'Best Parameters: {result.x}')
