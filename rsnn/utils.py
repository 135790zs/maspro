import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
import numpy as np
from config import cfg
rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


# def initialize_neurons():
#     N = {}
#     N['V'] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"],)) * cfg["eqb"]

#     if cfg["neuron"] == "ALIF":
#         N['U'] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"],)) * cfg["thr"]
#     elif cfg["neuron"] in ["Izhikevich", "LIF"]:
#         N['U'] = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))

#     N['Z'] = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))
#     N['H'] = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))
#     N['TZ'] = np.zeros(shape=(cfg["N_Rec"], cfg["N_R"],))

#     return N


# def initialize_weights():
#     W = {}

#     rng = np.random.default_rng()
#     W['W'] = rng.random(size=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))  # * 2 - 1
#     W['W'] *= cfg["W_mp"]

#     for r in range(cfg["N_Rec"]-1):
#         W['W'][r, :, :] = drop_weights(W=W['W'][r, :, :], recur_lay1=(r > 0))

#     W['B'] = rng.random(size=(cfg["N_Rec"]-2, cfg["N_R"],))

#     W['L'] = np.zeros(shape=(cfg["N_Rec"]-2, cfg["N_R"],))

#     W['EVV'] = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))
#     W['EVU'] = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))
#     W['ET'] = np.zeros(shape=(cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,))

#     return W


def initialize_log():
    rng = np.random.default_rng()
    neuron_shape = (cfg["Epochs"],) + (cfg["N_Rec"], cfg["N_R"],)
    weight_shape = (cfg["Epochs"],) + (cfg["N_Rec"]-1, cfg["N_R"]*2, cfg["N_R"]*2,)
    feedback_shape = (cfg["Epochs"], cfg["N_Rec"]-2, cfg["N_R"],)
    M = {
        "V": np.zeros(shape=neuron_shape),
        "U": np.zeros(shape=neuron_shape),
        "Z": np.zeros(shape=neuron_shape),
        "TZ": np.ones(shape=neuron_shape) * -cfg["dt_refr"],
        "H": np.zeros(shape=neuron_shape),
        "EVV": np.zeros(shape=weight_shape),
        "EVU": np.zeros(shape=weight_shape),
        "ET": np.zeros(shape=weight_shape),
        "W": rng.random(size=weight_shape),
        "B": np.ones(shape=feedback_shape),# * rng.random(),
        "L": np.ones(shape=feedback_shape),
        "input": np.zeros(shape=(cfg["Epochs"], cfg["N_I"])),
        "input_spike": np.zeros(shape=(cfg["Epochs"], cfg["N_I"])),
        "output": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
        "output_EMA": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
        "target": np.zeros(shape=(cfg["Epochs"], cfg["N_O"])),
        "target_EMA": np.zeros(shape=(cfg["Epochs"], cfg["N_O"]))
    }

    for r in range(cfg["N_Rec"]-1):
        M['W'][0, r, :, :] = drop_weights(W=M['W'][0, r, :, :],
                                          recur_lay1=(r > 0))
        M['W'][0, 0, :, :] = np.asarray([[0., 0.],
                                         [1000., 0.]])
        M['W'][0, 1, :, :] = np.asarray([[0., 0.],
                                         [0., 0.]])

    print("W", M['W'][0, ...])

    return M


def plot_logs(M, X, title=None):
    [M.pop(key) for key in ["TZ", "L"]]

    fig = plt.figure(constrained_layout=False)
    gsc = fig.add_gridspec(nrows=len(M)+1, ncols=1, hspace=0)
    axs = []
    labelpad = 15
    fontsize = 14
    fontsize_legend = 12

    if title:
        fig.suptitle(title, fontsize=20)

    lookup = {
        "V":   {"dim": 2, "label": "v^t"},
        "U":   {"dim": 2, "label": "u^t"},
        "Z":   {"dim": 2, "label": "z^t"},
        "X":   {"dim": 2, "label": "x^t"},
        "TZ":  {"dim": 2, "label": "TZ^t"},
        "EVV": {"dim": 3, "label": "\\epsilon^t"},
        "EVU": {"dim": 3, "label": "\\epsilon^t"},
        "W":   {"dim": 3, "label": "W^t"},
        "H":   {"dim": 2, "label": "h^t"},
        "ET":  {"dim": 3, "label": "e^t"},
    }

    # PRINT INPUTS
    axs.append(fig.add_subplot(gsc[len(axs), :],
                               sharex=axs[0] if axs else None))
    h = 0.5  # height of vlines
    rng = np.random.default_rng()
    inp_ys = rng.random(size=X.shape[0])
    for n_idx in [0, 1]:
        axs[-1].vlines(x=[idx for idx, val in
                          enumerate(X[:, n_idx]) if val],
                       ymin=n_idx+inp_ys/(1+h),
                       ymax=n_idx+(inp_ys+h)/(1+h),
                       colors=f'C{n_idx}',
                       linewidths=0.25,
                       label=f"$x_{n_idx}$")
    axs[-1].set_ylabel("$x^t_j$",
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)

    for key, arr in M.items():
        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))

        if arr.ndim == 2:
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


def eprop(model, M, X, t, uses_weights, r=None):
    Iz = np.dot(M['W'], M['Z']) if uses_weights else np.zeros(
            shape=M['Z'].shape)  # TODO: Weird what order of W, Z should be. Seems diff for units and rsnn
    Iz += X# if X.ndim == 2 else X
    if r == 0:
        print(M)
        print(Iz)

    if model == "LIF":
        M['Z'] = np.where(np.logical_and(t - M['TZ'] >= cfg["dt_refr"],
                                         M['V'] >= cfg["thr"]),
                          1,
                          0)
    elif model == "ALIF":
        M['Z'] = np.where(np.logical_and(t - M['TZ'] >= cfg["dt_refr"],
                                         M['V'] >= (cfg["thr"]
                                                    + cfg["beta"] * M['U'])),
                          1,
                          0)
    elif model == "Izhikevich":
        M['Z'] = np.where(M['V'] >= cfg["thr"], 1., 0.)

    M['TZ'] = np.where(M['Z'], t, M['TZ'])

    if model in ["LIF", "ALIF"]:
        R = (t - M['TZ'] == cfg["dt_refr"]).astype(int)

        M['V'] = (cfg["alpha"] * M['V']
                  + Iz - M['Z'] * cfg["alpha"] * M['V']
                  - R * cfg["alpha"] * M['V'])

    elif model == "Izhikevich":
        Vt = M['V'] - (M['V'] - cfg["eqb"]) * M['Z']
        Ut = M['U'] + cfg["refr1"] * M['Z']
        Vn = Vt + cfg["dt"] * (cfg["volt1"] * Vt**2
                               + cfg["volt2"] * Vt
                               + cfg["volt3"]
                               - Ut
                               + Iz)
        Un = Ut + cfg["dt"] * (cfg["refr2"] * Vt
                               - cfg["refr3"] * Ut)

    if model == "ALIF":
        M['U'] = cfg["rho"] * M['U'] + M['Z']

    if model in ["LIF", "ALIF"]:
        M['EVV'] = (cfg["alpha"] * (1 - M['Z'] - R) * M['EVV']
                    + M['Z'][np.newaxis].T)

    elif model == "Izhikevich":
        M['EVV'] = (M['EVV'] * (1 - M['Z']
                                + 2 * cfg["volt1"] * cfg["dt"] * M['V']
                                - 2 * cfg["volt1"] * cfg["dt"] * M['V']
                                    * M['Z']
                                + cfg["volt2"] * cfg["dt"]
                                - cfg["volt2"] * cfg["dt"] * M['Z'])
                    - M['EVU']  # Traub: "* cfg["dt"]" may have to be appended
                    + M['Z'][np.newaxis].T * cfg["dt"])
        M['EVU'] = (cfg["refr2"] * cfg["dt"] * M['EVV'] * (1 - M['Z'])
                    + M['EVU'] * (1 - cfg["refr3"] * cfg["dt"]))

    if model == "LIF":
        M['H'] = np.where(t - M['TZ'] < cfg["dt_refr"],
                          -cfg["gamma"],
                          cfg["gamma"] * np.clip(a=1 - (abs(M['V']
                                                        - cfg["thr"])
                                                        / cfg["thr"]),
                                                 a_min=0,
                                                 a_max=None))
    elif model == "ALIF":
        M['H'] = np.where(t - M['TZ'] < cfg["dt_refr"],
                          -cfg["gamma"],
                          cfg["gamma"] * np.clip(
                              a=1 - (abs(M['V'] - (cfg["thr"]
                                                   + cfg["beta"] * M['U']))
                                     / cfg["thr"]),
                              a_min=0,
                              a_max=None))
        M['EVU'] = M['H'] * M['EVV'] + (cfg["rho"]
                                        - M['H'] * cfg["beta"]) * M['EVU']
    elif model == "Izhikevich":
        M['V'] = Vn
        M['U'] = Un
        M['H'] = cfg["gamma"] * np.exp((np.clip(M['V'],
                                                a_min=None,
                                                a_max=cfg["thr"]) - cfg["thr"])
                                       / cfg["thr"])

    if model in ["LIF", "Izhikevich"]:
        M['ET'] = M['H'] * M['EVV']
    elif model == "ALIF":
        M['ET'] = M['H'] * (M['EVV'] - cfg["beta"] * M['EVU'])

    if M['L'].shape[-1] == M['ET'].shape[-1]:
        M['ET'] = M['ET'] * M['L']
    else:
        M['ET'] = M['ET'] * np.repeat(a=M['L'], repeats=2)

    D = np.where(M['W'], M['ET'], 0)  # only update nonzero
    M['W'] = M['W'] + D

    return M


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


def plot_drsnn(fig, gsc, M, ep, layers=(0, 1), neurons=(0, 0)):

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
            "V":   {"dim": 2, "label": "v^t"},
            "U":   {"dim": 2, "label": "u^t"},
            "Z":   {"dim": 2, "label": "z^t"},
            "X":   {"dim": 2, "label": "x^t"},
            "EVV": {"dim": 3, "label": "\\epsilon^t"},
            "EVU": {"dim": 3, "label": "\\epsilon^t"},
            "W":   {"dim": 3, "label": "W^t"},
            "H":   {"dim": 2, "label": "h^t"},
            "ET":  {"dim": 3, "label": "e^t"},
        }

        keyidx = 0
        for key, arr in M.items():
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
                inp_ys = rng.random(size=M["X"].shape[0])
                for n_idx in [0, 1]:
                    axs.vlines(x=[idx for idx, val in
                                  enumerate(M["X"][:ep+1, n_idx]) if val],
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
                axs.plot(arr1[:ep+1, neurons[0]],  # TODO: ep+1?
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
            axs.imshow(unflatten(M['V'][ep, r, :num]),
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
            axs.imshow(M['W'][ep, r, :, :cfg["N_R"]],
                       cmap='coolwarm',
                       vmin=0, vmax=100,
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
            axs.imshow(unflatten(M['Z'][ep, r, :num]),
                       cmap='gray',
                       vmin=0, vmax=1,
                       interpolation='nearest')

    if cfg["plot_io"]:
        # Output, target, error
        axs = fig.add_subplot(gsc[0, 3])
        axs.set_title(f"Input + spike")
        axs.plot(M["input"][:ep+1, :])
        axs.plot(M["input_spike"][:ep+1, :])
        axs = fig.add_subplot(gsc[1, 3])
        axs.set_title(f"Output + EMA")
        axs.plot(M["output"][:ep+1, :])
        axs.plot(M["output_EMA"][:ep+1, :])
        axs = fig.add_subplot(gsc[2, 3])
        axs.set_title(f"Target + EMA")
        axs.plot(M["target"][:ep+1, :])
        axs.plot(M["target_EMA"][:ep+1, :])
        axs = fig.add_subplot(gsc[3, 3])
        axs.set_title(f"Error + EMA")
        axs.plot(errfn(M["target"][:ep+1, :], M["output"][:ep+1, :]))
        axs.plot(errfn(M["target_EMA"][:ep+1, :], M["output_EMA"][:ep+1, :]))

    plt.draw()

    plt.pause((0.8 if ep == 1 else 0.01))
    fig.clf()

    return fig, gsc
