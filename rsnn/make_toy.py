import numpy as np
import matplotlib.pyplot as plt
from config import cfg

TYPE = "STDP"

for tvt_type in ["train", "val", "test"]:
    n_examples = cfg['n_examples'][tvt_type]
    S_len = cfg['maxlen']
    Si = 2
    So = 2
    plotidx = 0

    rng = np.random.default_rng()
    inp = np.zeros(shape=(n_examples, S_len, Si))
    tar = np.zeros(shape=(n_examples, S_len, So))

    for s in range(n_examples):
        if TYPE == "XOR":
            A = (1, 1,)
            dur = (rng.integers(1, 4), rng.integers(1, 4),)
            gap = (rng.integers(1, 5), rng.integers(1, 5),)
            off = (rng.integers(4), rng.integers(4),)
        elif TYPE == "STDP":
            int1 = rng.integers(8, 14)
            offset2 = rng.integers(3, int1//2)

        for t in range(S_len):
            if TYPE == "XOR":
                for n in range(Si):
                    inp[s, t, n] = (A[n] * ((t - off[n])
                                            % (dur[n] + gap[n]) < dur[n]))
            elif TYPE == "STDP":
                inp[s, t, 0] = t % int1 == 0
                inp[s, t, 1] = (t-offset2) % int1 == 0
        for t in range(S_len):  # classidx is XOR
            if TYPE == "XOR":
                for n in range(So):
                    ix = int(1-abs(inp[s, t, 0] - inp[s, t, 1]))
                    tar[s, t, ix] = 1
            elif TYPE == "STDP":
                tar[s, t, 0] = 1
                tar[s, t, 1] = 0

    np.save(f'{cfg["wavs_fname"]}_{tvt_type}.npy', inp)
    np.save(f'{cfg["phns_fname"]}_{tvt_type}.npy', tar)

fig = plt.figure(constrained_layout=False, figsize=(10, 5))
gsc = fig.add_gridspec(nrows=2, ncols=1, hspace=0.15)
axs = [fig.add_subplot(gsc[r, :]) for r in range(2)]

axs[0].imshow(inp[plotidx].T, aspect='auto', cmap='gray')
axs[1].imshow(tar[plotidx].T, aspect='auto', cmap='gray')

plt.savefig(f"../vis/example{TYPE}.pdf",
            bbox_inches='tight')

plt.close()

print("Made XOR dataset!")