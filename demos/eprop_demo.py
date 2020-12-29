import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams as rc

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'

nsteps = 1000
dt_ref = 20
thr = 7
alpha = 0.99
kappa = 0.5
A = 0.08
B = 0.10
C = 0.2

allvars = ["I", "V", "H", "TZ", "Z", "EV", "ET", "ETbar", "L", "DW"]
plotvars = ["V", "Z", "EV", "H", "ET", "DW"]
M = {}
rng = np.random.default_rng()

for var in allvars:
    M[var] = np.zeros(shape=(nsteps, 2))

for t in range(nsteps):
    M['I'][t] = rng.random(size=(2,)) / 10
    if t < nsteps * 0.45:
        M["I"][t, 0] = A
        M["I"][t, 1] = B
        if M["TZ"][t, 0] < M["TZ"][t, 1]:
            M["I"][t, 0] = C
    else:
        M["I"][t, 0] = B
        M["I"][t, 1] = A
        if M["TZ"][t, 1] < M["TZ"][t, 0]:
            M["I"][t, 1] = C

    M["Z"][t] = np.where(np.logical_and(
        M["V"][t] >= thr,
        t - M["TZ"][t] >= dt_ref), 1, 0)

    M['TZ'][t:, M['Z'][t]==1] = t

    M['H'][t] = np.where(t - M['TZ'][t] < dt_ref,
        0,
        0.3 * np.clip(
            a=1-(abs(M['V'][t] - thr)/thr),
            a_min=0,
            a_max=None))

    M["ET"][t] = M['H'][t] * M['EV'][t]

    M["ETbar"][t] = kappa * (M["ETbar"][t-1] if t else 0) + M["ET"][t]

    M["L"][t] = 1
    M["DW"][t] = M["L"][t] * ((M["DW"][t-1] if t else 0)
                              + M["ETbar"][t])

    if t != nsteps - 1:
        # V
        M["V"][t+1] = (alpha * M["V"][t]
                       + M["I"][t]
                       - M["Z"][t] * thr)

        M["EV"][t+1] = alpha * M["EV"][t] + np.flip(M["Z"][t])

fig = plt.figure(constrained_layout=False, figsize=(8, 6))
gsc = fig.add_gridspec(nrows=len(plotvars),
                       ncols=1,
                       hspace=0.075,
                       wspace=0.5)
axs = []

lookup = {
    "V": "$v^t_j$",
    "Z": "$z^t$",
    "ET": "$e_{{ji}}^t$",
    "ETbar": "$\\bar{{e}}_{{ji}}^t$",
    "EV": "$\\epsilon_{{ji}}^t$",
    "H": "$\\psi_j^t$",
    "DW": "$\\Delta W_{{ji}}^t$",
}

for var in plotvars:

    axs.append(fig.add_subplot(gsc[len(axs), :]))
    arr = M[var] if var in ["Z",] else M[var][:, 0]
    axs[-1].plot(arr)
    axs[-1].grid()

    axs[-1].set_ylabel(lookup[var],
                       rotation=0,
                       fontsize=17,
                       labelpad=20)

    axs[-1].set_xlabel("$t$", fontsize=20)

plt.savefig(f"../vis/eprop_demo_bellec.pdf",
            bbox_inches='tight')

plt.close()
