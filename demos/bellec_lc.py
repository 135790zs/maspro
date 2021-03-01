import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("~/code/eligibility_propagation/Figure_2_TIMIT/curve.csv")
np = df.to_numpy().T

labelpad = 38
fontsize = 8
fig = plt.figure(constrained_layout=False, figsize=(4, 3))
gsc = fig.add_gridspec(nrows=len(df.columns),
                       ncols=1, hspace=0.05)
axs = []

for arridx, arr in enumerate(np):
    name = df.columns[arridx]
    axs.append(fig.add_subplot(gsc[len(axs), :]))
    axs[-1].grid()
    axs[-1].plot(arr[arr >= 0])
    if name in ['Cross-entropy', 'Percentage wrong', 'Error (reg)']:
        axs[-1].set_yscale('log')

    axs[-1].set_ylabel(name,
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)
    axs[-1].tick_params(axis='both', which='major', labelsize=fontsize*0.8)
    axs[-1].tick_params(axis='both', which='minor', labelsize=fontsize*0.8)
    axs[-1].set_xlabel("Iteration", fontsize=fontsize)

plt.savefig(f"../vis/bellecLC.pdf",
            bbox_inches='tight')

plt.close()
