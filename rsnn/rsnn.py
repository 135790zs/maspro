import matplotlib.pyplot as plt
import numpy as np
from config import cfg
import utils as ut
from task import task1

plot_interval = 10

# Variable arrays
Nv = np.ones(shape=(cfg["N_Rec"], cfg["N_R"],)) * cfg["eqb"]

if cfg["neuron"] == "ALIF":
    Nu = np.ones(shape=(cfg["N_Rec"], cfg["N_R"],)) * cfg["thr"]
elif cfg["neuron"] in ["Izhikevich", "LIF"]:
    Nu = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))

Nz = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))
H = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))
TZ = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))

rng = np.random.default_rng()
W = rng.random(size=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))# * 2 - 1
W *= 1

for r in range(cfg["N_Rec"]-1):
    W[r, :, :] = ut.drop_weights(W=W[r, :, :], recur_lay1=(r > 0))

B = rng.random(size=(cfg["N_Rec"]-2, cfg["N_R"],))

L = np.zeros(shape=(cfg["N_Rec"]-2, cfg["N_R"],))

EVv = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))
EVu = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))
ET = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))

log = {
    "Nv": np.zeros(shape=(cfg["Epochs"],) + Nv.shape),
    "Nu": np.zeros(shape=(cfg["Epochs"],) + Nu.shape),
    "Nz": np.zeros(shape=(cfg["Epochs"],) + Nz.shape),
    "H": np.zeros(shape=(cfg["Epochs"],) + H.shape),
    "EVv": np.zeros(shape=(cfg["Epochs"],) + EVv.shape),
    "EVu": np.zeros(shape=(cfg["Epochs"],) + EVu.shape),
    "ET": np.zeros(shape=(cfg["Epochs"],) + ET.shape),
    "W": np.zeros(shape=(cfg["Epochs"],) + W.shape),
    "input": np.zeros(shape=(cfg["Epochs"], cfg["N_I"])),
    "input_spike": np.zeros(shape=(cfg["Epochs"], cfg["N_I"])),
    "output": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
    "output_EMA": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
    "target": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
    "target_EMA": np.zeros(shape=(cfg["Epochs"], cfg["N_O"]))
}

fig = plt.figure(constrained_layout=False)
gsc = fig.add_gridspec(nrows=max(8, 2 * cfg["N_Rec"] - 1), ncols=4, hspace=0.2)

plt.ion()

for ep in range(0, cfg["Epochs"]):

    # input is nonzero for first layer
    log["input"][ep, :] = task1(io_type="I", t=ep)
    # Bernoulli distribtion
    log["input_spike"][ep, :] = rng.binomial(n=1, p=log["input"][ep, :])

    # Feed to input layer R0
    Nv[0, :cfg["N_I"]] = np.where(log["input_spike"][ep, :],
                                  cfg["thr"] if cfg["neuron"] != "ALIF" else Nu[0, :cfg["N_I"]],
                                  Nv[0, :cfg["N_I"]])

    for r in range(0, cfg["N_Rec"] - 1):

        if cfg["neuron"] == "Izhikevich":
            Nvr, Nur, Nzr, EVv[r, :, :], EVu[r, :, :], Hr, W[r, :, :], \
                ET[r, :, :], TZr = ut.izh_eprop(
                    Nv=np.concatenate((Nv[r, :], Nv[r+1, :])),
                    Nu=np.concatenate((Nu[r, :], Nu[r+1, :])),
                    Nz=np.concatenate((Nz[r, :], Nz[r+1, :])),
                    TZ=np.concatenate((TZ[r, :], TZ[r+1, :])),
                    EVv=EVv[r, :, :],
                    EVu=EVu[r, :, :],
                    W=W[r, :, :],
                    L=L[r-1, :] if r > 0 else np.zeros(shape=cfg["N_R"]),
                    X=np.pad(array=log["input_spike"][ep, :],
                             pad_width=(0, 2*cfg["N_R"]-cfg["N_I"])),
                    t=ep)
        elif cfg["neuron"] == "ALIF":
            Nvr, Nur, Nzr, EVv[r, :, :], EVu[r, :, :], Hr, W[r, :, :], \
                ET[r, :, :], TZr = ut.alif_eprop(
                    Nv=np.concatenate((Nv[r, :], Nv[r+1, :])),
                    Nu=np.concatenate((Nu[r, :], Nu[r+1, :])),
                    Nz=np.concatenate((Nz[r, :], Nz[r+1, :])),
                    TZ=np.concatenate((TZ[r, :], TZ[r+1, :])),
                    EVv=EVv[r, :, :],
                    EVu=EVu[r, :, :],
                    W=W[r, :, :],
                    L=L[r-1, :] if r > 0 else np.zeros(shape=cfg["N_R"]),
                    X=np.pad(array=log["input_spike"][ep, :],
                             pad_width=(0, 2*cfg["N_R"]-cfg["N_I"])),
                    t=ep)
        elif cfg["neuron"] == "LIF":
            Nvr, Nzr, EVv[r, :, :], Hr, W[r, :, :], ET[r, :, :], TZr \
                = ut.lif_eprop(
                    Nv=np.concatenate((Nv[r, :], Nv[r+1, :])),
                    Nz=np.concatenate((Nz[r, :], Nz[r+1, :])),
                    TZ=np.concatenate((TZ[r, :], TZ[r+1, :])),
                    EVv=EVv[r, :, :],
                    W=W[r, :, :],
                    L=L[r-1, :] if r > 0 else np.zeros(shape=cfg["N_R"]),
                    X=np.pad(array=log["input_spike"][ep, :],
                             pad_width=(0, 2*cfg["N_R"]-cfg["N_I"])),
                    t=ep)

        Nv[r, :] = Nvr[:cfg["N_R"]]
        Nv[r+1, :] = Nvr[cfg["N_R"]:]
        if cfg["neuron"] not in ["LIF"]:
            Nu[r, :] = Nur[:cfg["N_R"]]
            Nu[r+1, :] = Nur[cfg["N_R"]:]
        Nz[r, :] = Nzr[:cfg["N_R"]]
        Nz[r+1, :] = Nzr[cfg["N_R"]:]
        TZ[r, :] = TZr[:cfg["N_R"]]
        TZ[r+1, :] = TZr[cfg["N_R"]:]
        H[r, :] = Hr[:cfg["N_R"]]
        H[r+1, :] = Hr[cfg["N_R"]:]

        log["Nv"][ep, :, :] = Nv
        log["Nu"][ep, :, :] = Nu
        log["Nz"][ep, :, :] = Nz
        log["H"][ep, :, :] = H
        log["ET"][ep, :, :, :] = ET
        log["EVv"][ep, :, :, :] = EVv
        log["EVu"][ep, :, :, :] = EVu
        log["W"][ep, :, :, :] = W

        X = np.zeros(shape=(cfg["N_R"],))  # First layer passed, set input to 0

    # W = W / np.real(np.max(np.linalg.eigvals(W))) * cfg["synscale"]

    log["output"][ep, :] = Nz[-1, :cfg["N_O"]]

    log["target"][ep, :] = task1(io_type="O", t=ep)

    if ep == 0:
        log["output_EMA"][ep, :] = log["output"][ep, :]
        log["target_EMA"][ep, :] = log["target"][ep, :]
    else:
        log["output_EMA"][ep, :] = (
            cfg["EMA"] * log["output"][ep, :]
            + (1 - cfg["EMA"]) * log["output_EMA"][ep-1, :])
        log["target_EMA"][ep, :] = (
            cfg["EMA"] * log["target"][ep, :]
            + (1 - cfg["EMA"]) * log["target_EMA"][ep-1, :])

    error = np.mean(ut.errfn(log["output_EMA"][:ep+1, :],
                             log["target_EMA"][:ep+1, :]),
                    axis=0)
    L = error * B

    if plot_interval and (ep % plot_interval == 0):
        fig, gsc = ut.plot_drsnn(fig=fig,
                                 gsc=gsc,
                                 Nv=Nv,
                                 W=W,
                                 Nz=Nz,
                                 log=log,
                                 ep=ep,
                                 layers=(0, 1),
                                 neurons=(0, 0))


# Merge e-prop--functions
# TODO: Implement adaptive e-prop
# TODO: Combine drsnn plot and plot_logs
# TODO: Find out if Bellec uses synscaling
# TODO: Implement Bellec TIMIT with ALIF
