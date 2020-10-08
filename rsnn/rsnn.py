import matplotlib.pyplot as plt
import numpy as np
from config import cfg
import utils as ut
from scipy.stats import poisson
import time

plot_interval = 1

# Variable arrays
Nv = np.ones(shape=(cfg["N_Rec"]+2, cfg["N_R"],)) * cfg["eqb"]
Nu = np.zeros(shape=(cfg["N_Rec"]+2, cfg["N_R"],))
Nz = np.zeros(shape=(cfg["N_Rec"]+2, cfg["N_R"],))
H = np.zeros(shape=(cfg["N_Rec"]+2, cfg["N_R"],))
TZ = np.zeros(shape=(cfg["N_Rec"]+2, cfg["N_R"],))

rng = np.random.default_rng()
W = rng.random(size=(cfg["N_Rec"]+1, cfg["N_R"]*2, cfg["N_R"]*2,)) * 2 - 1

for r in range(cfg["N_Rec"]+1):
    W[r, :, :] = ut.drop_weights(W=W[r, :, :], recur_lay1=(r > 0))

EVv = np.zeros(shape=(cfg["N_Rec"]+1, cfg["N_R"]*2, cfg["N_R"]*2,))
EVu = np.zeros(shape=(cfg["N_Rec"]+1, cfg["N_R"]*2, cfg["N_R"]*2,))
ET = np.zeros(shape=(cfg["N_Rec"]+1, cfg["N_R"]*2, cfg["N_R"]*2,))

log = {
    "Nv": np.zeros(shape=(cfg["Epochs"],) + Nv.shape),
    "Nu": np.zeros(shape=(cfg["Epochs"],) + Nu.shape),
    "Nz": np.zeros(shape=(cfg["Epochs"],) + Nz.shape),
    "H": np.zeros(shape=(cfg["Epochs"],) + H.shape),
    "EVv": np.zeros(shape=(cfg["Epochs"],) + EVv.shape),
    "EVu": np.zeros(shape=(cfg["Epochs"],) + EVu.shape),
    "ET": np.zeros(shape=(cfg["Epochs"],) + ET.shape),
    "W": np.zeros(shape=(cfg["Epochs"],) + W.shape)
}

fig = plt.figure(constrained_layout=False)
gsc = fig.add_gridspec(nrows=10, ncols=cfg["N_Rec"]+2, hspace=0.2)

plt.ion()

for ep in range(0, cfg["Epochs"]):

    dat = rng.random(size=(cfg["N_I"],)) * 0.1  # input is nonzero for first layer
    dp = np.random.binomial(n=1, p=dat)

    Nv[0, :cfg["N_I"]] = np.where(dp, cfg["thr"], Nv[0, :cfg["N_I"]])

    for r in range(0, cfg["N_Rec"]):

        Nvr, Nur, Nzr, EVv[r, :, :], EVu[r, :, :], Hr, W[r, :, :], \
            ET[r, :, :], TZr = ut.izh_eprop(
                Nv=np.concatenate((Nv[r, :], Nv[r+1, :])),
                Nu=np.concatenate((Nu[r, :], Nu[r+1, :])),
                Nz=np.concatenate((Nz[r, :], Nz[r+1, :])),
                TZ=np.concatenate((TZ[r, :], TZ[r+1, :])),
                H=np.concatenate((H[r, :], H[r+1, :])),
                EVv=EVv[r, :, :],
                EVu=EVu[r, :, :],
                ET=ET[r, :, :],
                W=W[r, :, :],
                X=np.pad(array=dp, pad_width=(0, 2*cfg["N_R"]-dp.shape[0])),
                t=ep)

        Nv[r, :] = Nvr[:cfg["N_R"]]
        Nu[r, :] = Nur[:cfg["N_R"]]
        Nz[r, :] = Nzr[:cfg["N_R"]]
        TZ[r, :] = TZr[:cfg["N_R"]]
        H[r, :] = Hr[:cfg["N_R"]]

        Nv[r+1, :] = Nvr[cfg["N_R"]:]
        Nu[r+1, :] = Nur[cfg["N_R"]:]
        Nz[r+1, :] = Nzr[cfg["N_R"]:]
        TZ[r+1, :] = TZr[cfg["N_R"]:]
        H[r+1, :] = Hr[cfg["N_R"]:]

        log["Nv"][ep, :, :] = Nv
        log["Nu"][ep, :, :] = Nu
        log["Nz"][ep, :, :] = Nz
        log["H"][ep, :, :] = H
        log["ET"][ep, :, :, :] = ET
        log["EVv"][ep, :, :, :] = EVv
        log["EVu"][ep, :, :, :] = EVu
        log["W"][ep, :, :, :] = W

        if plot_interval and (ep % plot_interval == 0 or ep == 0):
            fig, gsc = ut.plot_drsnn(fig=fig,
                                     gsc=gsc,
                                     Nv=Nv,
                                     W=W,
                                     log=log,
                                     ep=ep,
                                     layers=(0, 0),
                                     neurons=(0, 1))

        X = np.zeros(shape=(cfg["N_R"],))  # First layer passed, set input to 0

# TODO: Add Poisson
# TODO: Refactor ALIF
# TODO: Implement L
# TODO: Find out if Bellec uses synscaling
# TODO: Implement Bellec TIMIT with ALIF
