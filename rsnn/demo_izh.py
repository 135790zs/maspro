import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams as rc
rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'

np.random.seed(1)

gamma = 0.3
A = 4
B = 6
r = 0

thr = 30

reset = -65

dt = 0.25

mode = 'onoff'
# FloatType a = 0.02,
# FloatType b = 0.2,
# FloatType c = -65,
# FloatType d = 8,

nsteps = int(1000 / dt)
kappa = 0.9 * dt

allvars_n = ["I", "V", "Vt", "At", 'dvdv', 'dvdu', 'dudv', 'dudu', "A", "H", "Z"]
allvars_w = ["VV", "VA", "ET", "ETbar", "DW", 'W']
allvars_t = ["TZ"]
plotvars = ["V", "A", "Z", "H", 'VV', "VA", 'ET', 'W']

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
    if mode == 'onoff':
        if t > nsteps * 0.5:
            # pass
            M["I"][t, j] = A + np.random.random()*r
            if M["TZ"][i] < M["TZ"][j]:
                M["I"][t, i] = (B + np.random.random()*r)# * (t - M["TZ"][j])
        else:
            # pass
            M["I"][t, i] = A + np.random.random()*r
            if M["TZ"][j] < M["TZ"][i]:
                M["I"][t, j] = (B + np.random.random()*r)# * (t - M["TZ"][i])
    elif mode == 'cont':
        M['I'][t, i] = np.random.random() * r
        M['I'][t, j] = np.random.random() * r

    a = 0.02
    b = 0.2
    correct = False
    # Default: follow voltage and self
    # If blue spike: pure VA
    # If orng spike: +1
                 # * (1 + (5 + 0.08 * M['V'][t-1, j]) * dt) * M['VV'][t-1] \
                 # * (2.75 + 0.04 * M['V'][t-1, j]) * M['VV'][t-1] \
    M['VV'][t] = (1 - M['Z'][t-1, j]) \
                 * (0.7 + 1 * max(0, M['V'][t-1, j])) * M['VV'][t-1] \
                 - dt * M['VA'][t-1] \
                 + dt * M['Z'][t-1, i]
    # Default: follow VV
    # If blue spike: decay
    M['VA'][t] = (1 - M['Z'][t-1, j]) * dt * a * b * M['VV'][t-1] \
                 + (1 - dt * a) * M['VA'][t-1]

    if correct:
        M['VA'][t] = np.clip(M['VA'][t], -0.005, 0.005)
        M['VV'][t] = np.clip(M['VV'][t], -3., 3.)


    M['Vt'][t] = M['V'][t-1] - (M['V'][t-1] + 65) * M['Z'][t-1]
    M['At'][t] = M['A'][t-1] + 2 * M['Z'][t-1]
    M['V'][t] = M['Vt'][t] + dt * (0.04 * M['Vt'][t] ** 2 + 5 * M['Vt'][t] + 140 - M['At'][t] + M['I'][t])
    M['A'][t] = M['At'][t] + dt * (0.004 * M['Vt'][t] - 0.02 * M['At'][t])
    M['Z'][t] = np.where(M['V'][t] >= 30, 1, 0)


    M['TZ'] = np.where(M['Z'][t] == 1, t, M['TZ'])
    M['H'][t] = gamma * np.exp((np.clip(M['V'][t], None, 30) - 30) / 30)
    M['ET'][t] = M['H'][t, j] * M['VV'][t]

    M['ETbar'][t] = kappa * M['ETbar'][t-1] + M['ET'][t]
    M['DW'][t] = 0.01 * M['ETbar'][t]
    M['W'][t] = M['W'][t-1] + M['DW'][t]

fig = plt.figure(constrained_layout=False, figsize=(20, 12))
gsc = fig.add_gridspec(nrows=len(plotvars),
                       ncols=1,
                       hspace=0.15,
                       wspace=0.5)
axs = []

lookup = {
    "V":     "$v^t_j$",
    "dvdv":  "$dvdv$",
    "dvdu":  "$dvdu$",
    "dudv":  "$dudv$",
    "dudu":  "$dudu$",
    "A":     "$a^t_j$",
    "At":     "$\\tilde{{a}}^t_j$",
    "Vt":     "$\\tilde{{v}}^t_j$",
    "Z":     "$z^t$",
    "I":     "$I^t$",
    "ET":    "$e_{{ji}}^t$",
    "ETbar": "$\\bar{{e}}_{{ji}}^t$",
    "VV":    "$\\epsilon_{{ji, v}}^t$",
    "VV1":   "$\\epsilon1_{{ji, v}}^t$",
    "VV2":   "$\\epsilon2_{{ji, v}}^t$",
    "VV3":   "$\\epsilon3_{{ji, v}}^t$",
    "VA":   "$\\epsilon_{{ji, a}}^t$",
    "VA1":   "$\\epsilon1_{{ji, a}}^t$",
    "VA2":   "$\\epsilon2_{{ji, a}}^t$",
    "H":     "$\\psi_j^t$",
    "DW":    "$\\Delta W_{{ji}}^t$",
    "W":    "$W_{{ji}}^t$",
}

for var in plotvars:
    axs.append(fig.add_subplot(gsc[len(axs), :]))
    arr = M[var] if var not in ['H', 'V', 'A'] else M[var][:, 1]
    if var in ['Z', 'I']:
        arr = np.flip(arr, axis=1)
        axs[-1].plot(arr[:, 0], linewidth=0.7, label='$z^t_j$')
        axs[-1].plot(arr[:, 1], linewidth=0.7, label='$z^t_i$')
        plt.legend(fontsize=11, labelspacing=0.2)
    else:
        axs[-1].plot(arr, linewidth=0.7)
    if var in ['VV', 'VA']:
        axs[-1].set_yscale('symlog')
    axs[-1].grid()

    axs[-1].set_ylabel(lookup[var],
                       rotation=0,
                       fontsize=17,
                       labelpad=20)

    axs[-1].set_xlabel("$t$", fontsize=20)

plt.savefig(f"../vis/demo_izh{'_corrected' if correct else ''}.pdf",
            bbox_inches='tight')

plt.close()
