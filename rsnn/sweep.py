from rsnn import run_rsnn
from config import cfg
from scipy.optimize import minimize
import itertools

x0 = [  # Linear hyperparameters
    128,  # W_mp
]

y0 = [  # Integer hyperparameters
    [4, 8, 16, 32, 64, 128, 256, 512],  # N_R
    # [2],        # N_Rec
]


def rsnn_aux(x, y):
    cfg0 = cfg

    cfg0["W_mp"] = x[0]
    cfg0["N_R"] = y[0]
    # cfg0["N_Rec"] = y[1]
    error = run_rsnn(cfg=cfg0)
    print(error, end='\r')
    return error


for y in list(itertools.product(*y0)):
    res = minimize(rsnn_aux, x0, args=(y,), method="Nelder-Mead", tol=1e-6, options={'maxiter': 200})
    print()
    print(y)
    print(res)
