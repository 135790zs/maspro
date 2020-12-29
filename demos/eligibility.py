import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams as rc

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'

nsteps = 200
dt_ref = 20
thr = 7
alpha = 0.98
kappa = 0.5
A = 0.08
B = 0.10
C = 0.2

allvars = ["ET", "ET1", "ET2", "L", "DW"]
plotvars = ["ET", "L", "DW"]
M = {}
rng = np.random.default_rng()

for var in allvars:
    M[var] = np.zeros(shape=(nsteps, ))


for t in range(nsteps):
    s = 20
    s2 = 60
    s3 = 120
    if t < s:
        M['ET1'][t] = 0
    else:
        M['ET1'][t] = (t-s) * 2 / np.exp((t-s+1)/10)

    if t < s3:
        M['ET2'][t] = 0
    else:
        M['ET2'][t] = - (t-s3) * 2 / np.exp((t-s3+1)/5)

    M['ET'][t] = M['ET1'][t] + M['ET2'][t]
    M['L'][t] = 10/t if t >= s2 else 0
    M['DW'][t] = (M['DW'][t-1] if t else 7) + M['ET'][t] * M["L"][t]


fig = plt.figure(constrained_layout=False, figsize=(5, 2))
gsc = fig.add_gridspec(nrows=len(plotvars),
                       ncols=1,
                       hspace=0.2,
                       wspace=0.5)
axs = []

lookup = {
    "ET": "$e_{{ji}}^t$",
    "L": "$L_j^t$",
    "DW": "$w_{{ji}}^t$",
}

for var in plotvars:
    arr = M[var]
    axs.append(fig.add_subplot(gsc[len(axs), :]))
    axs[-1].plot(arr, color='k')
    # axs[-1].set_ylim(None, np.max(arr)*1.3)
    axs[-1].set_ylabel(lookup[var],
                       rotation=0,
                       fontsize=13,
                       labelpad=20)

    # Hide grid lines
    axs[-1].grid(False)

    # Hide axes ticks
    axs[-1].set_xticks([])
    axs[-1].set_yticks([])
    axs[-1].spines["top"].set_visible(False)
    axs[-1].spines["right"].set_visible(False)
    axs[-1].spines["left"].set_visible(False)
    axs[-1].spines["bottom"].set_visible(False)
    axs[-1].axhline(y=0, color='k', linewidth=0.4)
    axs[-1].axvline(x=s2, color='k', dashes=(2,2))
    axs[-1].axvline(x=s3, color='k', dashes=(2,2))

axs[-1].set_xlabel("$t$", fontsize=13)

plt.savefig(f"../vis/eligibility.pdf",
            bbox_inches='tight')

plt.close()
