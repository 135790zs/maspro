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
        "Nv": np.zeros(shape=(T, num,)),
        "Nu": np.zeros(shape=(T, num,)),
        "X": X,
        "Nz": np.zeros(shape=(T, num,)),
        "H": np.zeros(shape=(T, num,)),
        "EVv": np.zeros(shape=(T, num, num,)),
        "EVu": np.zeros(shape=(T, num, num,)),
        "ET": np.zeros(shape=(T, num, num,)),
        "W": np.zeros(shape=(T, num, num,)),
    }

    # Variable arrays
    if model == "Izhikevich":
        Nv = np.ones(shape=(num,)) * cfg["eqb"]
    elif model in ["LIF", "ALIF"]:
        Nv = np.zeros(shape=(num,))
    if model == "ALIF":
        Nu = np.ones(shape=(num,)) * cfg["thr"]
    elif model in ["LIF", "Izhikevich"]:
        Nu = np.zeros(shape=(num,))

    Nz = np.zeros(shape=(num,))
    H = np.zeros(shape=(num,))
    TZ = np.zeros(shape=(num,))

    rng = np.random.default_rng()
    W = rng.random(size=(num, num,))
    np.fill_diagonal(W, 0.)

    EVv = np.zeros(shape=(num, num,))
    EVu = np.zeros(shape=(num, num,))
    ET = np.zeros(shape=(num, num,))

    for t in range(0, T):

        Nv, Nu, Nz, EVv, EVu, H, W, ET, TZ = ut.eprop(
            model=model, Nv=Nv, Nu=Nu, Nz=Nz, X=X, EVv=EVv, EVu=EVu,
            W=W, TZ=TZ, t=t, uses_weights=uses_weights, L=None)

        log["Nv"][t, :] = Nv
        log["Nu"][t, :] = Nu
        log["Nz"][t, :] = Nz
        log["H"][t, :] = H
        log["EVv"][t, :, :] = EVv
        log["EVu"][t, :, :] = EVu
        log["ET"][t, :, :] = ET
        log["W"][t, :, :] = W

    ut.plot_logs(log, title=f"{model} e-prop")


# simulate_neurons(model="LIF")
# simulate_neurons(model="ALIF", uses_weights=True)
simulate_neurons(model="Izhikevich", uses_weights=True)
