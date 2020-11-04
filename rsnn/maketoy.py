import os
import numpy as np
from config import cfg

n_examples = 42
S_len = cfg['maxlen']
Si = 2
So = 4

rng = np.random.default_rng()
inp = np.zeros(shape=(n_examples, S_len, Si))
tar = np.zeros(shape=(n_examples, S_len, So))
for s in range(n_examples):
	A = (1, 1,)
	dur = (rng.integers(4), rng.integers(4),)
	gap = (rng.integers(5), rng.integers(5),)
	off = (rng.integers(4), rng.integers(4),)
	for t in range(S_len):
	    for n in range(Si):
	        inp[s, t, n] = A[n] * ((t - off[n]) % (dur[n] + gap[n]) < dur[n])
	for t in range(S_len):  # classidx is sum of active
	    for n in range(So):
	        ix = int(inp[s, t, 0] + 2 * inp[s, t, 1])
	        tar[s, t, ix] = 1
for s in range(n_examples):
	for t in range(S_len):
		print(f"S{s}:\tinp: {inp[s, t]}: tar: {tar[s, t]}")

np.save(cfg["wavs_fname"], inp)
np.save(cfg["phns_fname"], tar)