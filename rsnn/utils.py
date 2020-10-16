import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
import numpy as np
from config import cfg
rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


def initialize_neurons():
    N = {}
    N['V'] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"],)) * cfg["eqb"]

    if cfg["neuron"] == "ALIF":
        N['U'] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"],)) * cfg["thr"]
    elif cfg["neuron"] in ["Izhikevich", "LIF"]:
        N['U'] = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))

    N['Z'] = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))
    N['H'] = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))
    N['TZ'] = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))

    return N


def initialize_weights():
    W = {}

    rng = np.random.default_rng()
    W['W'] = rng.random(size=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))  # * 2 - 1
    W['W'] *= cfg["W_mp"]

    for r in range(cfg["N_Rec"]-1):
        W['W'][r, :, :] = drop_weights(W=W['W'][r, :, :], recur_lay1=(r > 0))

    W['B'] = rng.random(size=(cfg["N_Rec"]-2, cfg["N_R"],))

    W['L'] = np.zeros(shape=(cfg["N_Rec"]-2, cfg["N_R"],))

    W['EVV'] = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))
    W['EVU'] = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))
    W['ET'] = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))

    return W


def initialize_log():
    neuron_shape = (cfg["Epochs"],) + (cfg["N_Rec"], cfg["N_R"],)
    weight_shape = (cfg["Epochs"],) + (cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,)
    return {
        "V": np.zeros(shape=neuron_shape),
        "U": np.zeros(shape=neuron_shape),
        "Z": np.zeros(shape=neuron_shape),
        "H": np.zeros(shape=neuron_shape),
        "EVV": np.zeros(shape=weight_shape),
        "EVU": np.zeros(shape=weight_shape),
        "ET": np.zeros(shape=weight_shape),
        "W": np.zeros(shape=weight_shape),
        "input": np.zeros(shape=(cfg["Epochs"], cfg["N_I"])),
        "input_spike": np.zeros(shape=(cfg["Epochs"], cfg["N_I"])),
        "output": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
        "output_EMA": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
        "target": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
        "target_EMA": np.zeros(shape=(cfg["Epochs"], cfg["N_O"]))
    }


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
        "V":  {"label": "v^t"},
        "U":  {"label": "u^t"},
        "Z":  {"label": "z^t"},
        "X":   {"label": "x^t"},
        "EVV": {"label": "\\epsilon^t"},
        "EVU": {"label": "\\epsilon^t"},
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


def eprop(model, X, t, TZ, V, Z, EVV, L, W, uses_weights=True, U=None, EVU=None):
    I = np.dot(W, Z) if uses_weights else np.zeros(shape=Z.shape)
    I += X[t, :] if X.ndim == 2 else X

    print(V[0])

    if model == "LIF":
        Z = np.where(np.logical_and(t - TZ >= cfg["dt_refr"],
                                    V >= cfg["thr"]),
                     1,
                     0)
    elif model == "ALIF":
        Z = np.where(np.logical_and(t - TZ >= cfg["dt_refr"],
                                    V >= (cfg["thr"] + cfg["beta"] * U)),
                     1,
                     0)
    elif model == "Izhikevich":
        Z = np.where(V >= cfg["thr"], 1., 0.)

    TZ = np.where(Z, t, TZ)

    if model in ["LIF", "ALIF"]:
        R = (t - TZ == cfg["dt_refr"]).astype(int)

        V = (cfg["alpha"] * V
             + I - Z * cfg["alpha"] * V
             - R * cfg["alpha"] * V)

    elif model == "Izhikevich":
        Vt = V - (V - cfg["eqb"]) * Z
        Ut = U + cfg["refr1"] * Z
        Vn = Vt + cfg["dt"] * (cfg["volt1"] * Vt**2
                               + cfg["volt2"] * Vt
                               + cfg["volt3"]
                               - Ut
                               + I)
        Un = Ut + cfg["dt"] * (cfg["refr2"] * Vt
                               - cfg["refr3"] * Ut)

    if model == "ALIF":
        U = cfg["rho"] * U + Z

    if model in ["LIF", "ALIF"]:
        EVV = cfg["alpha"] * (1 - Z - R) * EVV + Z[np.newaxis].T

    elif model == "Izhikevich":
        EVV = (EVV * (1 - Z
                      + 2 * cfg["volt1"] * cfg["dt"] * V
                      - 2 * cfg["volt1"] * cfg["dt"] * V * Z
                      + cfg["volt2"] * cfg["dt"]
                      - cfg["volt2"] * cfg["dt"] * Z)
               - EVU  # Traub: "* cfg["dt"]" may have to be appended
               + Z[np.newaxis].T * cfg["dt"])
        EVU = (cfg["refr2"] * cfg["dt"] * EVV * (1 - Z)
               + EVU * (1 - cfg["refr3"] * cfg["dt"]))

    if model == "LIF":
        H = np.where(t - TZ < cfg["dt_refr"],
                     -cfg["gamma"],
                     cfg["gamma"] * np.clip(a=1 - (abs(V - cfg["thr"])
                                                   / cfg["thr"]),
                                            a_min=0,
                                            a_max=None))
    elif model == "ALIF":
        H = np.where(t - TZ < cfg["dt_refr"],
                     -cfg["gamma"],
                     cfg["gamma"] * np.clip(
                        a=1 - (abs(V - (cfg["thr"] + cfg["beta"] * U))
                               / cfg["thr"]),
                        a_min=0,
                        a_max=None))
        EVU = H * EVV + (cfg["rho"] - H * cfg["beta"]) * EVU
    elif model == "Izhikevich":
        V = Vn
        U = Un
        H = cfg["gamma"] * np.exp((np.clip(V,
                                           a_min=None,
                                           a_max=cfg["thr"]) - cfg["thr"])
                                  / cfg["thr"])

    if model in ["LIF", "Izhikevich"]:
        ET = H * EVV
    elif model == "ALIF":
        ET = H * (EVV - cfg["beta"] * EVU)

    if L is not None:
        ET = ET * np.repeat(a=L, repeats=2)

    W = W + np.where(W, ET, 0)  # only update nonzero weights

    N = {
        'V': V,
        'U': U,
        'Z': Z,
        'H': H,
        'TZ': TZ,
    }

    W = {
        'W': W,
        'EVV': EVV,
        'EVU': EVU,
        'ET': ET,
    }

    return N, W


def drop_weights(W, recur_lay1=True):
    N = W.shape[0]//2

    if recur_lay1:
        np.fill_diagonal(W[:N, :N], 0)  # Zero diag NW: no self-connections
    else:
        W[:N, :N] = 0  # empty NW: don't recur input layer

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


def errfn(a1, a2):
    return np.sum(np.abs(a1 - a2), axis=1)


def plot_drsnn(fig, gsc, V, W, Z, log, ep, layers=(0, 1), neurons=(0, 0)):

    assert layers[1] - layers[0] == 0 or layers[0] - layers[1] == -1

    # If to next layer, weight matrix is appended after recurrent.
    n1 = neurons[1] + (cfg["N_R"] if layers[0] != layers[1] else 0)

    fig.suptitle(f"Epoch {ep}/{cfg['Epochs']}, "
                 f"$N_{{({layers[0]}), {neurons[0]}}}$ and "
                 f"$N_{{({layers[1]}), {neurons[1]}}}$; "
                 f"$W_{{({layers[0]}), {neurons[0]}, {n1}}}$ and"
                 f"$W_{{({layers[0]}), {n1}, {neurons[0]}}}$", fontsize=20)

    labelpad = 15
    fontsize = 14
    fontsize_legend = 12
    if cfg["plot_pair"]:

        lookup = {
            "V":  {"dim": 2, "label": "v^t"},
            "U":  {"dim": 2, "label": "u^t"},
            "Z":  {"dim": 2, "label": "z^t"},
            "X":   {"dim": 2, "label": "x^t"},
            "EVV": {"dim": 3, "label": "\\epsilon^t"},
            "EVU": {"dim": 3, "label": "\\epsilon^t"},
            "W":   {"dim": 3, "label": "W^t"},
            "H":   {"dim": 2, "label": "h^t"},
            "ET":  {"dim": 3, "label": "e^t"},
        }

        keyidx = 0
        for key, arr in log.items():
            if key not in lookup.keys():
                continue
            if arr.ndim != lookup[key]["dim"]:
                arr1 = arr[:ep+1, layers[0], ...]
                arr2 = arr[:ep+1, layers[1], ...]
            else:  # If singlelayer
                arr1 = arr

            axs = fig.add_subplot(gsc[keyidx, 0])
            keyidx += 1

            if key == "X":
                h = 0.5  # height of vlines
                rng = np.random.default_rng()
                inp_ys = rng.random(size=log["X"].shape[0])
                for n_idx in [0, 1]:
                    axs.vlines(x=[idx for idx, val in
                                  enumerate(log["X"][:ep+1, n_idx]) if val],
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
                axs.plot(arr1[:ep+1, neurons[0]],
                         label=f"${lookup[key]['label']}_0$")
                axs.plot(arr2[:ep+1, neurons[0]],
                         label=f"${lookup[key]['label']}_1$")
                axs.set_ylabel(f"${lookup[key]['label']}_j$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

            elif arr1.ndim == 3:  # Weights etc
                EVtype = key[2:]+',' if key[:2] == "EV" else ""
                axs.plot(arr1[:ep+1, neurons[0], n1],
                         label=f"${lookup[key]['label']}_{{{EVtype}{neurons[0]}{n1}}}$")
                axs.plot(arr2[:ep+1, n1, neurons[0]],
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

    if cfg["plot_heatmaps"]:
        # Neuron heatmaps
        for r in range(0, cfg["N_Rec"]):
            num = cfg["N_I"] if r == 0 \
                else cfg["N_O"] if r == cfg["N_Rec"] \
                else cfg["N_R"]
            axs = fig.add_subplot(gsc[r, 1])
            axs.set_ylabel(f"$v_{{{r}, i}}$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)
            axs.imshow(unflatten(V[r, :num]),
                       cmap='coolwarm',
                       vmin=-85, vmax=cfg["eqb"],
                       interpolation='nearest')

        # Weight heatmaps
        for r in range(0, cfg["N_Rec"]-1):
            axs = fig.add_subplot(gsc[2*r+cfg["N_Rec"]:2*r+2+cfg["N_Rec"], 1])
            axs.set_ylabel(f"$W_{{{r}, i, j}}$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)
            axs.imshow(W[r, :, :cfg["N_R"]],
                       cmap='coolwarm',
                       vmin=-cfg["W_mp"], vmax=cfg["W_mp"],
                       interpolation='nearest')

        # Neuron spikes
        for r in range(0, cfg["N_Rec"]):
            num = cfg["N_I"] if r == 0 \
                else cfg["N_O"] if r == cfg["N_Rec"] \
                else cfg["N_R"]
            axs = fig.add_subplot(gsc[r, 2])
            axs.set_ylabel(f"$z_{{{r}, i}}$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)
            axs.imshow(unflatten(Z[r, :num]),
                       cmap='gray',
                       vmin=0, vmax=1,
                       interpolation='nearest')

    if cfg["plot_io"]:
        # Output, target, error
        axs = fig.add_subplot(gsc[0, 3])
        axs.set_title(f"Input + spike")
        axs.plot(log["input"][:ep+1, :])
        axs.plot(log["input_spike"][:ep+1, :])
        axs = fig.add_subplot(gsc[1, 3])
        axs.set_title(f"Output + EMA")
        axs.plot(log["output"][:ep+1, :])
        axs.plot(log["output_EMA"][:ep+1, :])
        axs = fig.add_subplot(gsc[2, 3])
        axs.set_title(f"Target + EMA")
        axs.plot(log["target"][:ep+1, :])
        axs.plot(log["target_EMA"][:ep+1, :])
        axs = fig.add_subplot(gsc[3, 3])
        axs.set_title(f"Error + EMA")
        axs.plot(errfn(log["target"][:ep+1, :], log["output"][:ep+1, :]))
        axs.plot(errfn(log["target_EMA"][:ep+1, :], log["output_EMA"][:ep+1, :]))

    plt.draw()

    plt.pause((0.8 if ep == 1 else 0.01))
    fig.clf()

    return fig, gsc
