from config import cfg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
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
                   - 2 * cfg["volt1"] * cfg["dt"] * Nv * Nz  # not sure about Nvp here
                   + cfg["volt2"] * cfg["dt"]
                   - cfg["volt2"] * cfg["dt"] * Nz)
            - EVu * cfg["dt"]
            + Nz[np.newaxis].T * cfg["dt"])


def EVu_next(EVv, EVu, Nz):
    return (EVv * (cfg["refr2"] * cfg["dt"]
                   - cfg["refr2"] * cfg["dt"] * Nz)
            + EVu * (1
                     - cfg["refr3"] * cfg["dt"]))


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
