import numpy as np
import matplotlib.pyplot as plt
from config import cfg

n_examples = cfg['n_toy_examples']
S_len = cfg['maxlen']
Si = 2
So = 2
plotidx = 2

rng = np.random.default_rng()
inp = np.zeros(shape=(n_examples, S_len, Si))
tar = np.zeros(shape=(n_examples, S_len, So))
for s in range(n_examples):
    A = (1, 1,)
    dur = (rng.integers(1, 4), rng.integers(1, 4),)
    gap = (rng.integers(1, 5), rng.integers(1, 5),)
    off = (rng.integers(4), rng.integers(4),)
    for t in range(S_len):
        for n in range(Si):
            inp[s, t, n] = A[n] * ((t - off[n]) % (dur[n] + gap[n]) < dur[n])
    for t in range(S_len):  # classidx is XOR
        for n in range(So):
            ix = int(1-abs(inp[s, t, 0] - inp[s, t, 1]))
            tar[s, t, ix] = 1

print(f"Saved {n_examples} with {S_len} frames each. "
	  f"Each frame has {Si} inputs and {So} targets.")
print(f"Saved with shapes {inp.shape} and {tar.shape}!")

np.save(cfg["wavs_fname"], inp)
np.save(cfg["phns_fname"], tar)


fig = plt.figure(constrained_layout=False, figsize=(10, 5))
gsc = fig.add_gridspec(nrows=2, ncols=1, hspace=0.15)
axs = [fig.add_subplot(gsc[r, :]) for r in range(2)] 

axs[0].imshow(inp[plotidx].T, aspect='auto', cmap='gray')
axs[1].imshow(tar[plotidx].T, aspect='auto', cmap='gray')
# axs[0].set_axis('off')
# axs[1].axis('off')
plt.savefig(f"../vis/exampleTOY.pdf",
            bbox_inches='tight')

plt.close()