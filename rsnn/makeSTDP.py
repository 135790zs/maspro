import numpy as np
import matplotlib.pyplot as plt
from config import cfg

for tvt_type in ["train", "val", "test"]:
    n_examples = cfg['n_examples'][tvt_type]
    S_len = cfg['maxlen']
    Si = 2
    So = 1
    plotidx = 0

    rng = np.random.default_rng()
    inp = np.zeros(shape=(n_examples, S_len, Si))
    tar = np.zeros(shape=(n_examples, S_len, So))

    for s in range(n_examples):
        int1 = rng.integers(8, 14)
        offset2 = rng.integers(2, int1//2)

        for t in range(S_len):
            inp[s, t, 0] = t % int1 == 0
            inp[s, t, 1] = (t-offset2) % int1 == 0
        for t in range(S_len):  # classidx is XOR
            tar[s, t, 0] = 1 if t < S_len//2 else 0

    np.save(f'{cfg["wavs_fname"]}_{tvt_type}.npy', inp)
    np.save(f'{cfg["phns_fname"]}_{tvt_type}.npy', tar)

fig = plt.figure(constrained_layout=False, figsize=(10, 5))
gsc = fig.add_gridspec(nrows=2, ncols=1, hspace=0.15)
axs = [fig.add_subplot(gsc[r, :]) for r in range(2)]

axs[0].imshow(inp[plotidx].T, aspect='auto', cmap='gray')
axs[1].imshow(tar[plotidx].T, aspect='auto', cmap='gray')

plt.savefig(f"../vis/exampleSTDP.pdf",
            bbox_inches='tight')

plt.close()

print("Made STDP dataset!")
