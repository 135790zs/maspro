import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
import time
import numpy as np
from config import cfg
rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


def U_tilde(Nu, Nz):
    return Nu + cfg["refr1"] * Nz


def V_tilde(Nv, Nz):
    return Nv - (Nv - cfg["eqb"]) * Nz


def V_next(Nv, Nu, Nz, I):
    Nvt = V_tilde(Nv=Nv, Nz=Nz)
    Nut = U_tilde(Nu=Nu, Nz=Nz)

    return (Nvt + cfg["dt"] * (cfg["volt1"] * Nvt**2
                               + cfg["volt2"] * Nvt
                               + cfg["volt3"]
                               - Nut
                               + I))


def U_next(Nu, Nz, Nv):
    Nvt = V_tilde(Nv=Nv, Nz=Nz)
    Nut = U_tilde(Nu=Nu, Nz=Nz)

    return (Nut + cfg["dt"] * (cfg["refr2"] * Nvt
                               - cfg["refr3"] * Nut))


def EVv_next(EVv, EVu, Nz, Nv):
    return (EVv * (1 - Nz
                   + 2 * cfg["volt1"] * cfg["dt"] * Nv
                   - 2 * cfg["volt1"] * cfg["dt"] * Nv * Nz
                   + cfg["volt2"] * cfg["dt"]
                   - cfg["volt2"] * cfg["dt"] * Nz)
            - EVu  # Traub: "* cfg["dt"]" may have to be appended
            + Nz[np.newaxis].T * cfg["dt"])


def EVu_next(EVv, EVu, Nz):
    return (cfg["refr2"] * cfg["dt"] * EVv * (1 - Nz)
            + EVu * (1 - cfg["refr3"] * cfg["dt"]))


def H_next(Nv):
    return cfg["gamma"] * np.exp((np.clip(Nv,
                                          a_min=None,
                                          a_max=cfg["thr"]) - cfg["thr"])
                                 / cfg["thr"])


def plot_logs(log, title=None):
    fig = plt.figure(constrained_layout=False)
    gsc = fig.add_gridspec(nrows=len(log), ncols=1, hspace=0)
    axs = []
    labelpad = 15
    fontsize = 14
    fontsize_legend = 12

    if title:
        fig.suptitle(title, fontsize=20)

    lookup = {
        "Nv":  {"label": "v^t"},
        "Nu":  {"label": "u^t"},
        "Nz":  {"label": "z^t"},
        "X":   {"label": "x^t"},
        "EVv": {"label": "\\epsilon^t"},
        "EVu": {"label": "\\epsilon^t"},
        "W":   {"label": "W^t"},
        "H":   {"label": "h^t"},
        "ET":  {"label": "e^t"},
    }

    for key, arr in log.items():
        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))

        if key == "X":
            h = 0.5  # height of vlines
            rng = np.random.default_rng()
            inp_ys = rng.random(size=log["X"].shape[0])
            for n_idx in [0, 1]:
                axs[-1].vlines(x=[idx for idx, val in
                                  enumerate(log["X"][:, n_idx]) if val],
                               ymin=n_idx+inp_ys/(1+h),
                               ymax=n_idx+(inp_ys+h)/(1+h),
                               colors=f'C{n_idx}',
                               linewidths=0.25,
                               label=f"$x_{n_idx}$")
            axs[-1].set_ylabel("$x^t_j$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        elif arr.ndim == 2:
            axs[-1].plot(arr[:, 0],
                         label=f"${lookup[key]['label']}_0$")
            axs[-1].plot(arr[:, 1],
                         label=f"${lookup[key]['label']}_1$")
            axs[-1].set_ylabel(f"${lookup[key]['label']}_j$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        elif arr.ndim == 3:
            EVtype = key[2:]+',' if key[:2] == "EV" else ""
            axs[-1].plot(arr[:, 0, 1],
                         label=f"${lookup[key]['label']}_{{{EVtype}0,1}}$")
            axs[-1].plot(arr[:, 1, 0],
                         label=f"${lookup[key]['label']}_{{{EVtype}1,0}}$")
            axs[-1].set_ylabel(f"${lookup[key]['label']}_{{{EVtype}i,j}}$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        axs[-1].legend(fontsize=fontsize_legend,
                       loc="upper right",
                       ncol=2)
        axs[-1].grid(linestyle='--')

    axs[-1].set_xlabel("$t$", fontsize=fontsize)

    plt.show()


def get_artificial_input(T, num, dur, diff, interval, val, switch_interval):
    X = np.zeros(shape=(T, num))
    for t in range(0, T):
        if t % (switch_interval*2) < switch_interval:
            X[t, 0] = val if t % interval <= dur else 0
            X[t, 1] = val if (t % interval <= diff+dur and t % interval > diff) \
                else 0
        else:
            X[t, 0] = val if (t % interval <= diff+dur and t % interval > diff) \
                else 0
            X[t, 1] = val if t % interval <= dur else 0
    return X


def izh_eprop(Nv, Nu, Nz, X, EVv, EVu, H, W, ET, TZ, t, uses_weights=True):

    I = np.dot(W, Nz) if uses_weights else np.zeros(shape=Nz.shape)
    I += X[t, :] if X.ndim == 2 else X

    Nz = np.where(Nv >= cfg["thr"], 1., 0.)
    TZ = np.where(Nv >= cfg["thr"], t, TZ)

    Nvn = V_next(Nu=Nu, Nz=Nz, Nv=Nv, I=I)
    Nun = U_next(Nu=Nu, Nz=Nz, Nv=Nv)

    # Should this operate on Nvn instead? Probably not..?
    EVvn = EVv_next(EVv=EVv, EVu=EVu, Nz=Nz, Nv=Nv)
    EVun = EVu_next(EVv=EVv, EVu=EVu, Nz=Nz)

    # What about this one? Probably both or neither.
    H = H_next(Nv=Nvn)

    ET = H * EVvn

    W = W + np.where(W, ET, 0)  # only update nonzero weights

    EVv = EVvn
    EVu = EVun
    Nv = Nvn
    Nu = Nun

    return Nv, Nu, Nz, EVv, EVu, H, W, ET, TZ


def drop_weights(W, recur_lay1=True):
    N = W.shape[0]//2

    if recur_lay1:
        np.fill_diagonal(W[:N, :N], 0)  # Zero diag NW: no self-connections
    else:
        W[:N, :N] = 0  # Zero full NW: don't recur input layer

    W[:, N:] = 0  # Zero full E: can't go back, nor recur next layer

    return W


def unflatten(arr):
    a = arr.flatten().shape[0]
    b = int(np.ceil(np.sqrt(a)))
    arr = np.concatenate((arr.flatten(), np.zeros(b*b-a)))
    arr = np.reshape(arr, (b, b))
    c = b
    while (b*c-a) >= b:
        c -= 1
        arr = arr[:-1, :]
    return arr


def normalize(arr):
    return np.interp(arr, (arr.min(), arr.max()), (-1, 1))


def plot_drsnn(fig, gsc, Nv, W, log, ep, layers=(0, 0), neurons=(0, 1)):

    start = time.time()
    assert layers[1] - layers[0] == 0 or layers[0] - layers[1] == -1
    n1 = neurons[1] + (cfg["N_R"] if layers[0] != layers[1] else 0)
    fig.suptitle(f"Epoch {ep}/{cfg['Epochs']}, "
                 f"$N_{{({layers[0]}), {neurons[0]}}}$ and "
                 f"$N_{{({layers[1]}), {neurons[1]}}}$; "
                 f"$W_{{({layers[0]}), {neurons[0]}, {n1}}}$ and"
                 f"$W_{{({layers[0]}), {n1}, {neurons[0]}}}$", fontsize=20)

    for r in range(0, cfg["N_Rec"]+2):

        num = cfg["N_I"] if r == 0 \
            else cfg["N_O"] if r == cfg["N_Rec"]+2 \
            else cfg["N_R"]
        axs = fig.add_subplot(gsc[0, r])
        axs.set_title(f"$v_{{{r}, i}}$")
        axs.imshow(unflatten(normalize(Nv[r, :num])),
                   cmap='coolwarm',
                   vmin=0, vmax=1,
                   interpolation='nearest')

    for r in range(0, cfg["N_Rec"]+1):
        axs = fig.add_subplot(gsc[1, r])
        axs.set_title(f"$W_{{{r}, i, j}}$")
        axs.imshow(W[r, :, :],
                   cmap='coolwarm',
                   vmin=-1, vmax=1,
                   interpolation='nearest')

    lookup = {
        "Nv":  {"dim": 2, "label": "v^t"},
        "Nu":  {"dim": 2, "label": "u^t"},
        "Nz":  {"dim": 2, "label": "z^t"},
        "X":   {"dim": 2, "label": "x^t"},
        "EVv": {"dim": 3, "label": "\\epsilon^t"},
        "EVu": {"dim": 3, "label": "\\epsilon^t"},
        "W":   {"dim": 3, "label": "W^t"},
        "H":   {"dim": 2, "label": "h^t"},
        "ET":  {"dim": 3, "label": "e^t"},
    }

    labelpad = 15
    fontsize = 14
    fontsize_legend = 12
    keyidx = 1

    for key, arr in log.items():
        print(f"Took {time.time()-start:.3f}s to plot in epoch {ep} {key}")
        if arr.ndim != lookup[key]["dim"]:
            arr1 = arr[:ep, layers[0], ...]
            arr2 = arr[:ep, layers[1], ...]
        else:  # If singlelayer
            arr1 = arr

        axs = fig.add_subplot(gsc[keyidx, -1])
        keyidx += 1

        if key == "X":
            h = 0.5  # height of vlines
            rng = np.random.default_rng()
            inp_ys = rng.random(size=log["X"].shape[0])
            for n_idx in [0, 1]:
                axs.vlines(x=[idx for idx, val in
                              enumerate(log["X"][:ep, n_idx]) if val],
                           ymin=n_idx+inp_ys/(1+h),
                           ymax=n_idx+(inp_ys+h)/(1+h),
                           colors=f'C{n_idx}',
                           linewidths=0.25,
                           label=f"$x_{n_idx}$")
            axs.set_ylabel("$x^t_j$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)

        elif arr1.ndim == 2:  # Voltage etc
            axs.plot(arr1[:ep, neurons[0]],
                     label=f"${lookup[key]['label']}_0$")
            axs.plot(arr2[:ep, neurons[0]],
                     label=f"${lookup[key]['label']}_1$")
            axs.set_ylabel(f"${lookup[key]['label']}_j$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)

        elif arr1.ndim == 3:  # Weights etc
            EVtype = key[2:]+',' if key[:2] == "EV" else ""
            axs.plot(arr1[:ep, neurons[0], n1],
                     label=f"${lookup[key]['label']}_{{{EVtype}{neurons[0]}{n1}}}$")
            axs.plot(arr2[:ep, n1, neurons[0]],
                     label=f"${lookup[key]['label']}_{{{EVtype}{n1}{neurons[0]}}}$")
            axs.set_ylabel(f"${lookup[key]['label']}_{{{EVtype}i,j}}$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)
        if arr.ndim == lookup[key]["dim"]:
            axs.legend(fontsize=fontsize_legend,
                       loc="upper right",
                       ncol=2)
        axs.grid(linestyle='--')
    # axs[axidx].plot(logs["Nv"][:ep, 1, 0])
    # axs[axidx].plot(logs["Nv"][:ep, 2, 0])

    plt.draw()
    plt.pause(0.0001)

    fig.clf()

    print(f"Took {time.time()-start:.3f}s to plot in epoch {ep} C")

    return fig, gsc


# TODO: Combine drsnn plot and plot_logs
