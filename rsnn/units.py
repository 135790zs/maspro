import sys
import numpy as np
from matplotlib import rcParams as rc
from config import cfg
import utils as ut

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'
np.set_printoptions(threshold=sys.maxsize)


def simulate_neurons(model, T=1000, num=2, uses_weights=False):

    if model in ["LIF", "ALIF"]:
        X = ut.get_artificial_input(T=T,
                                    num=num,
                                    dur=18,
                                    diff=5,
                                    interval=100,
                                    val=3.55,
                                    switch_interval=500)

    elif model == "Izhikevich":
        X = ut.get_artificial_input(T=T,
                                    num=num,
                                    dur=30,
                                    diff=8,
                                    interval=100,
                                    val=32,
                                    switch_interval=500)

    # Logging arrays
    log = {  # Plotting will follow this order
        "V": np.zeros(shape=(T, num,)),
        "U": np.zeros(shape=(T, num,)),
        "X": X,
        "Z": np.zeros(shape=(T, num,)),
        "H": np.zeros(shape=(T, num,)),
        "EVV": np.zeros(shape=(T, num, num,)),
        "EVU": np.zeros(shape=(T, num, num,)),
        "ET": np.zeros(shape=(T, num, num,)),
        "W": np.zeros(shape=(T, num, num,)),
    }

    N = {}
    W = {}

    # Variable arrays
    if model == "Izhikevich":
        N['V'] = np.ones(shape=(num,)) * cfg["eqb"]
    elif model in ["LIF", "ALIF"]:
        N['V'] = np.zeros(shape=(num,))
    if model == "ALIF":
        N['U'] = np.ones(shape=(num,)) * cfg["thr"]
    elif model in ["LIF", "Izhikevich"]:
        N['U'] = np.zeros(shape=(num,))

    N['Z'] = np.zeros(shape=(num,))
    N['H'] = np.zeros(shape=(num,))
    N['TZ'] = np.zeros(shape=(num,))

    rng = np.random.default_rng()
    W['W'] = rng.random(size=(num, num,))
    np.fill_diagonal(W['W'], 0.)

    W['EVV'] = np.zeros(shape=(num, num,))
    W['EVU'] = np.zeros(shape=(num, num,))
    W['ET'] = np.zeros(shape=(num, num,))

    for t in range(0, T):
        Nt, Wt = ut.eprop(
            model=model,
            V=N['V'],
            U=N['U'],
            Z=N['Z'],
            X=X,
            EVV=W['EVV'],
            EVU=W['EVU'],
            W=W['W'],
            TZ=N['TZ'],
            t=t,
            uses_weights=uses_weights,
            L=None)

        for key, item in Nt.items():

            N[key] = item
            if key != "TZ":  # No TZ log exists
                log[key][t, :] = item

        for key, item in Wt.items():
            W[key] = item
            log[key][t, :, :] = item
        # log["V"][t, :] = Nv
        # log["U"][t, :] = Nu
        # log["Z"][t, :] = Nz
        # log["H"][t, :] = H
        # log["EVV"][t, :, :] = EVv
        # log["EVU"][t, :, :] = EVu
        # log["ET"][t, :, :] = ET
        # log["W"][t, :, :] = W

    ut.plot_logs(log, title=f"{model} e-prop")


# simulate_neurons(model="LIF")
# simulate_neurons(model="ALIF", uses_weights=True)
simulate_neurons(model="Izhikevich", uses_weights=True)
