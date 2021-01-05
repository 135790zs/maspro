import matplotlib.pyplot as plt
import numpy as np

for wtype in ["W", "W_out", "b_out", "B"]:

	fname = f"../vault/2021-01-04-16:58:53/checkpoints/{wtype}.npy"
	W = np.load(fname).flatten()
	W = W[W!=0]

	plt.hist(W, bins=(32 if wtype == "b_out" else 256))
	plt.title(wtype)
	plt.savefig(f"../vis/{wtype}_dist.pdf")
	plt.show()
