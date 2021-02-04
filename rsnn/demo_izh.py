import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams as rc
rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


gamma = 0.3
A = 5
B = 1

thr = 30

reset = -65

dt = 1
# FloatType a = 0.02,
# FloatType b = 0.2,
# FloatType c = -65,
# FloatType d = 8,

nsteps = int(1000 / dt)
kappa = 0.9 * dt

allvars_n = ["I", "V", 'dvdv', 'dvdu', 'dudv', 'dudu', "A", "H", "Z"]
allvars_w = ["VV", "VA", 'VV1', "VV2", 'VV3', 'VA1', 'VA2', "ET", "ETbar", "DW", 'W']
allvars_t = ["TZ"]
plotvars = ["I", "V", "A", "Z", 'VV1', "VV2", 'VV3', 'VV', 'VA1', 'VA2', "VA", "H", 'ET', 'W']

M = {}

for var in allvars_n:
    M[var] = np.zeros(shape=(nsteps, 2))
M['V'] = np.ones_like(M['V']) * -65

for var in allvars_w:
    M[var] = np.zeros(shape=(nsteps,))

for var in allvars_t:
    M[var] = np.zeros(shape=(2,))

i = 0
j = 1
for t in range(nsteps):

    if t > nsteps * 0.45:
        # pass
        M["I"][t, j] = A
        if M["TZ"][i] < M["TZ"][j]:
            M["I"][t, i] = B * (t - M["TZ"][j])
    else:
        # pass
        M["I"][t, i] = A
        if M["TZ"][j] < M["TZ"][i]:
            M["I"][t, j] = B * (t - M["TZ"][i])
    # M["I"][t, i] += np.random.random()*2
    # M["I"][t, j] += np.random.random()*2
    # TUNE THE 0.99's!!

    # Magnitude on orange spike
    M['VV1'][t] = (1 - M['Z'][t-1, j]) * (1 + (5 + 0.08 * M['V'][t-1, j]) * dt) * M['VV'][t-1]
    M['VV2'][t] = - dt * M['VA'][t-1]
    M['VV3'][t] = dt * M['Z'][t-1, i]
    M['VV'][t] = M['VV1'][t] + M['VV2'][t] + M['VV3'][t]

    # Magnitude on orange spike
    M['VA1'][t] = (1 - M['Z'][t-1, j]) * dt * 0.004 * M['VV'][t-1]
    # M['VA1'][t] = (1 - M['Z'][t-1, j]) * (2 * M['H'][t-1, j] + dt * 0.004 - dt * 0.04 * M['H'][t-1, j]) * M['VV'][t-1]
    M['VA2'][t] = (1 - dt * 0.02) * M['VA'][t-1]
    M['VA'][t] = (M['VA1'][t] + M ['VA2'][t])


    # M['A'][t] = M['A'][t-1] + np.where(M['V'][t-1] >= 30, 8, 0)
    # M['V'][t] = np.where(M['V'][t-1] >= 30, -65, M['V'][t-1])

    M['V'][t] += dt * (0.04 * M['V'][t]**2 + 5 * M['V'][t] + 140 - M['A'][t] + M['I'][t])
    M['A'][t] += dt * 0.02 * (0.2 * M['V'][t-1] - M['A'][t])

    M['Z'][t] = np.where(M['V'][t] >= 30, 1, 0)
    M['V'][t] = np.where(M['V'][t] > 30, 30, M['V'][t])

    M['TZ'] = np.where(M['Z'][t] == 1, t, M['TZ'])
    M['H'][t] = gamma * np.exp((np.clip(M['V'][t], None, 30) - 30) / 30)
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
    "VV1":   "$\\epsilon1_{{v, ji}}^t$",
    "VV2":   "$\\epsilon2_{{v, ji}}^t$",
    "VV3":   "$\\epsilon3_{{v, ji}}^t$",
    "VA":   "$\\epsilon_{{a, ji}}^t$",
    "VA1":   "$\\epsilon1_{{a, ji}}^t$",
    "VA2":   "$\\epsilon2_{{a, ji}}^t$",
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
