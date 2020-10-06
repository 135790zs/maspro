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
    return cfg["gamma"] * np.exp((np.clip(Nv, a_min=None, a_max=cfg["H1"]) - cfg["H1"])
                                 / cfg["H1"])


def plot_logs(log):
    fig = plt.figure(constrained_layout=False)
    gsc = fig.add_gridspec(nrows=9, ncols=1, hspace=0)
    axs = []
    labelpad = 15
    fontsize = 14
    fontsize_legend = 12

    axs.append(fig.add_subplot(gsc[len(axs), :]))
    axs[-1].plot(log["Nv"][:, 0], label="$v^t_0$")
    axs[-1].plot(log["Nv"][:, 1], label="$v^t_1$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$v^t_j$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
    h = 0.5  # height of vlines
    rng = np.random.default_rng()
    inp_ys = rng.random(size=log["X"].shape[0])
    for n_idx in [0, 1]:
        axs[-1].vlines(x=[idx for idx, val in enumerate(log["X"][:, n_idx]) if val],
                       ymin=n_idx+inp_ys/(1+h), ymax=n_idx+(inp_ys+h)/(1+h),
                       colors=f'C{n_idx}',
                       linewidths=0.25,
                       label=f"$x_{n_idx}$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$x^t_j$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
    axs[-1].plot(log["Nu"][:, 0], label="$u^t_0$")
    axs[-1].plot(log["Nu"][:, 1], label="$u^t_1$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$u^t_j$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
    axs[-1].plot(log["Nz"][:, 0], label="$z^t_0$")
    axs[-1].plot(log["Nz"][:, 1], label="$z^t_1$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$z^t_j$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
    axs[-1].plot(log["EVv"][:, 0, 1], label="$\\epsilon^t_{0,1,v}$")
    axs[-1].plot(log["EVv"][:, 1, 0], label="$\\epsilon^t_{1,0,v}$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$\\epsilon^t_{i,j,v}$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
    axs[-1].plot(log["EVu"][:, 0, 1], label="$\\epsilon^t_{0,1,u}$")
    axs[-1].plot(log["EVu"][:, 1, 0], label="$\\epsilon^t_{1,0,u}$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$\\epsilon^t_{i,j,u}$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
    axs[-1].plot(log["H"][:, 0], label="$h^t_0$")
    axs[-1].plot(log["H"][:, 1], label="$h^t_1$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$h_j^t$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
    axs[-1].plot(log["ET"][:, 0, 1], label="$e^t_{0,1}$")
    axs[-1].plot(log["ET"][:, 1, 0], label="$e^t_{1,0}$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$e^t_{i,j}$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
    axs[-1].plot(log["W"][:, 0, 1], label="$w_{0,1}$")
    axs[-1].plot(log["W"][:, 1, 0], label="$w_{1,0}$")
    axs[-1].legend(fontsize=fontsize_legend, loc="upper right", ncol=2)
    axs[-1].set_ylabel("$w^t_{i,j}$", rotation=0, labelpad=labelpad, fontsize=fontsize)

    axs[-1].set_xlabel("$t$", fontsize=fontsize)

    plt.show()

# def vtilde(v, z):
#     return v - (v + 65) * z


# def utilde(u, z):
#     return u + 2 * z


# def vnext(v, u, z, I):
#     vtil = vtilde(v=v, z=z)
#     return (vtil
#             + config["dt"]*((0.04*vtil**2)
#                             + 5*vtil
#                             + 140
#                             - utilde(u=u, z=z)
#                             + I))


# def unext(u, v, z):
#     util = utilde(u=u, z=z)
#     return (util
#             + config["dt"]*(0.004 * vtilde(v=v, z=z)
#                             - 0.02 * util))


# def h(v):
#     return config["gamma"] * np.exp((min(v, config["H1"]) - config["H1"])
#                                     / config["H1"])


# def evvnext(zi, zj, vi, vj, evv, evu):
#     # term1 = (1 - zj)*(1 + (config["EVV1"] * vj + config["EVV2"]) * config["dt"]) \
#     #          * evv
#     term1 = (1 - zj
#              + 0.08*config["dt"]*vj
#              - 0.08*config["dt"]*zj*vj
#              + 5*config["dt"]
#              - 5*config["dt"]*zj) * evv
#     term2 = - config["dt"] * evu
#     term3 = zi * config["dt"]
#     return term1 + term2 + term3


# def evunext(zi, zj, evv, evu):
#     term1 = 0.004 * config["dt"] * (1 - zj) * evv
#     term2 = (1 - 0.02 * config["dt"]) * evu
#     return term1 + term2


# # TODO NEXT: Implement from Bellec directly, see if Traub is wrong or my own fns
# # If latter: thoroughly check everything
