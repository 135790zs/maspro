import itertools
from scipy.optimize import minimize
from rsnn import main
from config import cfg

pr = {  # Linear hyperparameters
    "fraction_ALIF": 0.5,  # W_mp
}

pn = {  # Integer hyperparameters
    "N_R": [4, 8, 16, 32, 64, 128, 256, 512],  # N_R
    "eta": [1e-2, 1e-3, 1e-4],# [2],        # N_Rec
}


def rsnn_aux(pr, pnp):
    cfg0 = cfg
    print(pr)
    for k, v in pnp.items():
        print(k)

    error = main(cfg=cfg0)
    # print(f"{error} with \tx={xs}, \ty={ys}", end='\r')
    return error


init_guess = cfg

for pnp in list(itertools.product(*pn.values())):  # mix the integers, try all
    print(list(pnp))
    res = minimize(rsnn_aux,
                   list(pr.values()),  # Initial guess
                   args=(pnp,),  # Additional args passed
                   method="Nelder-Mead",
                   tol=1e-6,
                   options={'maxiter': 1})
    print()
    print(pnp)
    print(res)
    print()
    with open("log.txt", 'a') as f:
        f.write(f"{res.fun}\t")
        f.write(f"{pnp}\t")
        f.write(f"{res.x}\n")
