import itertools
from scipy.optimize import minimize
from rsnn import main
from config import cfg

x0 = [  # Linear hyperparameters
    128,  # W_mp
]

y0 = [  # Integer hyperparameters
    [4, 8, 16, 32, 64, 128, 256, 512],  # N_R
    # [2],        # N_Rec
]


def rsnn_aux(xs, ys):
    cfg0 = cfg

    error = main(cfg=cfg0)
    print(f"{error} with \tx={xs}, \ty={ys}", end='\r')
    return error


for y in list(itertools.product(*y0)):
    res = minimize(rsnn_aux,
                   x0,
                   args=(y,),
                   method="Nelder-Mead",
                   tol=1e-6,
                   options={'maxiter': 200})
    print()
    print(y)
    print(res)
