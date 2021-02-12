import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams as rc
rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'

RESET_FIX = False

nsteps = 300
dt_ref = 8
thr = 1.6
alpha = 0.8
kappa = 0.8
rho = 0.995
beta = 0.184

A = 0.5


allvars = ["I", "V", "A", "Z", "TZ"]

plotvars = ["I", "V", "A", "Z"]

M = {}
rng = np.random.default_rng()

for var in allvars:
    M[var] = np.zeros(shape=(nsteps,))

for t in range(nsteps):

    M["I"][t] = A + 0.05 * np.sin(t*0.025*np.pi)

    M["Z"][t] = np.where(
        np.logical_and(M["V"][t] >= (thr + beta * M["A"][t]),
                       t - M["TZ"][t] >= dt_ref),
        1,
        0)

    M['TZ'][t:, M['Z'][t] == 1] = t


    if t != nsteps - 1:
        M["A"][t+1] = rho * M["A"][t] + M["Z"][t]

        M["V"][t+1] = (alpha * M["V"][t]
                       + M["I"][t]
                       - M["Z"][t] * (alpha * M["V"][t] if RESET_FIX else thr)
                       - (alpha * M["V"][t] * (t - M["TZ"][t] == dt_ref)
                          if RESET_FIX else 0))


fig = plt.figure(constrained_layout=False, figsize=(8, 6))
gsc = fig.add_gridspec(nrows=len(plotvars),
                       ncols=1,
                       hspace=0.15,
                       wspace=0.5)
axs = []

lookup = {
    "V":     "$v^t_j$",
    "A":     "$a^t_j$",
    "Z":     "$z^t_j$",
    "I":     "$I^t_j$",
}

for var in plotvars:
    axs.append(fig.add_subplot(gsc[len(axs), :]))
    arr = M[var] if var in ["I", "V", "A", "Z"] else M[var][:, 0]
    axs[-1].plot(arr, linewidth=0.8)
    axs[-1].grid()

    axs[-1].set_ylabel(lookup[var],
                       rotation=0,
                       fontsize=17,
                       labelpad=20)

    axs[-1].set_xlabel("$t$", fontsize=20)

plt.savefig(f"../vis/simple{'stdp' if RESET_FIX else ''}"
            f"alif.pdf",
            bbox_inches='tight')

plt.close()
