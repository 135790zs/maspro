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
rho = 0.975
beta = 0.184
A = 0.25
B = 0.35
C = 0.45


allvars = ["I", "V", "A", "H", "TZ", "Z", "EV", "EVA", "ET", "ETbar", "L", "DW"]

plotvars = ["I", "V", "A", "Z", "H", "EV", "EVA", "ET", "DW"]

M = {}
rng = np.random.default_rng()

for var in allvars:
    M[var] = np.zeros(shape=(nsteps, 2))

for t in range(nsteps):

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

    M["Z"][t] = np.where(
        np.logical_and(M["V"][t] >= (thr + beta * M["A"][t]),
                       t - M["TZ"][t] >= dt_ref),
        1,
        0)

    M['TZ'][t:, M['Z'][t] == 1] = t

    M['H'][t] = np.where(
        t - M['TZ'][t] < dt_ref,
        - 0.3 if RESET_FIX else 0,
        0.3 * np.clip(
            a=1-(abs(M['V'][t] - (thr + beta * M['A'][t]))
                 / thr),
            a_min=0,
            a_max=None))

    M["ET"][t] = M['H'][t] * (M['EV'][t] - (beta * M["EVA"][t]))

    M["ETbar"][t] = kappa * (M["ETbar"][t-1] if t else 0) + M["ET"][t]

    M["L"][t] = 1
    M["DW"][t] = M["L"][t] * ((M["DW"][t-1] if t else 0)
                              + M["ETbar"][t])

    if t != nsteps - 1:
        M["A"][t+1] = rho * M["A"][t] + M["Z"][t]

        M["V"][t+1] = (alpha * M["V"][t]
                       + M["I"][t]
                       - M["Z"][t] * (alpha * M["V"][t] if RESET_FIX else thr)
                       - (alpha * M["V"][t] * (t - M["TZ"][t] == dt_ref)
                          if RESET_FIX else 0))
        M["EVA"][t+1] = (M['H'][t] * np.flip(M["EV"][t])
                         + (rho
                            - M['H'][t] * beta)
                         * M["EVA"][t])
            # IF ABOVE DOESN'T WORK FOR BELLEC, use:
            # M["EVA"][t+1] = (M["H"][t] * alpha * M["EVV"][t]
            #                  + M["H"][t] * np.flip(M["Z"][t])
            #                  + rho * M["EVA"][t]
            #                  - M["H"][t] * beta * M["EVA"][t])

        M["EV"][t+1] = (
            np.flip(M["Z"][t])   # Spike when orange
            + alpha * M["EV"][t] * (1 if not RESET_FIX else (
                1
                - M["Z"][t]  # Null when blue
                - (t - np.flip(M["TZ"][t]) <= dt_ref)  # Null=time after orange
                - (t - M["TZ"][t] == dt_ref))))  # Null=time after blue

fig = plt.figure(constrained_layout=False, figsize=(8, 6))
gsc = fig.add_gridspec(nrows=len(plotvars),
                       ncols=1,
                       hspace=0.15,
                       wspace=0.5)
axs = []

lookup = {
    "V":     "$v^t$",
    "A":     "$a^t$",
    "Z":     "$z^t$",
    "I":     "$I^t$",
    "ET":    "$e_{{ji}}^t$",
    "ETbar": "$\\bar{{e}}_{{ji}}^t$",
    "EV":    "$\\epsilon_{{v, ji}}^t$",
    "EVA":   "$\\epsilon_{{a, ji}}^t$",
    "H":     "$\\psi_j^t$",
    "DW":    "$\\Delta W_{{ji}}^t$",
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

plt.savefig(f"../vis/{'stdp' if RESET_FIX else ''}"
            f"alif.pdf",
            bbox_inches='tight')

plt.close()
