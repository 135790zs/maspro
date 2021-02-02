import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams as rc
rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


nsteps = 1000
# dt_ref = 20
# alpha = 0.99
kappa = 0.9
gamma = 0.3
# rho = 0.9975
# beta = 0.01
A = 5
B = 1

thr = 30

v1 = 0.04
v2 = 5
v3 = 140
a1 = 0.004
a2 = -0.02
reset = -65
psi = 30


allvars_n = ["I", "V", 'dvdv', 'dvdu', 'dudv', 'dudu', "A", "H", "Z"]
allvars_w = ["VV", "VA", "ET", "ETbar", "DW", 'W']
allvars_t = ["TZ"]
plotvars = ["I", "V", "A", "Z", "VV", "VA", "H", 'ET', 'W']

M = {}

for var in allvars_n:
    M[var] = np.zeros(shape=(nsteps, 2))

for var in allvars_w:
    M[var] = np.zeros(shape=(nsteps,))

for var in allvars_t:
    M[var] = np.zeros(shape=(2,))

i = 0
j = 1
for t in range(nsteps):

    if t > nsteps * 0.55:
        # pass
        M["I"][t, j] = A
        if M["TZ"][i] < M["TZ"][j]:
            M["I"][t, i] = B * (t - M["TZ"][j])
    else:
        # pass
        M["I"][t, i] = A
        if M["TZ"][j] < M["TZ"][i]:
            M["I"][t, j] = B * (t - M["TZ"][i])

    # If no postsynaptic spike, then VV follows V. Used to generate spike at presynaptic
    M['dvdv'][t] = (1 - M['Z'][t-1]) * 0.5 * (1 + v2 + 2 * v1 * M['V'][t-1])
    # sign of VV depends on sign of VA
    M['dvdu'][t] = -1
    # if no postsynaptic spike, then VA follows VV slightly
    M['dudv'][t] = a1 * (1 - M['Z'][t-1])
    # VA decay factor
    M['dudu'][t] = 1 + a2

    M['VV'][t] = M['dvdv'][t, j] * M['VV'][t-1] + M['dvdu'][t, j] * M['VA'][t-1] + M['Z'][t-1, i]
    M['VA'][t] = M['dudv'][t, j] * M['VV'][t-1] + M['dudu'][t, j] * M['VA'][t-1]

    vt = M['V'][t-1] - (M['V'][t-1] - reset) * M['Z'][t-1]
    at = M['A'][t-1] + 2 * M['Z'][t-1]

    M['V'][t] = vt + v1 * vt ** 2 + v2 * vt + v3 - at + M["I"][t]
    M['A'][t] = at + a1 * vt + a2 * at

    M['Z'][t] = np.where(M['V'][t] >= thr, 1, 0)

    M['TZ'] = np.where(M['Z'][t] == 1, t, M['TZ'])
    M['H'][t] = gamma * np.exp((np.clip(M['V'][t], None, psi) - psi) / psi)
    M['ET'][t] = M['H'][t, j] * M['VV'][t]

    M['ETbar'][t] = kappa * M['ETbar'][t-1] + M['ET'][t]
    M['DW'][t] = 0.01 * M['ETbar'][t]
    M['W'][t] = M['W'][t-1] + M['DW'][t]

fig = plt.figure(constrained_layout=False, figsize=(8, 6))
gsc = fig.add_gridspec(nrows=len(plotvars),
                       ncols=1,
                       hspace=0.075,
                       wspace=0.5)
axs = []

lookup = {
    "V":     "$v^t_j$",
    "dvdv":  "$dvdv$",
    "dvdu":  "$dvdu$",
    "dudv":  "$dudv$",
    "dudu":  "$dudu$",
    "A":     "$a^t_j$",
    "Z":     "$z^t$",
    "I":     "$I^t$",
    "ET":    "$e_{{ji}}^t$",
    "ETbar": "$\\bar{{e}}_{{ji}}^t$",
    "VV":    "$\\epsilon_{{v, ji}}^t$",
    "VA":   "$\\epsilon_{{a, ji}}^t$",
    "H":     "$\\psi_j^t$",
    "DW":    "$\\Delta W_{{ji}}^t$",
    "W":    "$W_{{ji}}^t$",
}

for var in plotvars:
    axs.append(fig.add_subplot(gsc[len(axs), :]))
    arr = M[var]
    # arr = arr[:, 0] if arr.ndim > 1 else arr
    axs[-1].plot(arr, linewidth=0.7)
    axs[-1].grid()

    axs[-1].set_ylabel(lookup[var],
                       rotation=0,
                       fontsize=17,
                       labelpad=20)

    axs[-1].set_xlabel("$t$", fontsize=20)

plt.savefig(f"../vis/demo_izh.pdf",
            bbox_inches='tight')

plt.close()
