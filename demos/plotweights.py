import matplotlib.pyplot as plt
import numpy as np

for wtype in ["W_in", "W_rec", "out", "bias", "B"]:
	fname = f"../log/2021-01-29-11:22:43/checkpoints/{wtype}.npy"

	W = np.load(fname)
	num_subnets = W.shape[0] if wtype != 'bias' else 1
	num_layers = W.shape[1] if wtype not in ['out', 'bias'] else 1

	for s in range(num_subnets):


		if wtype not in ["W_in", "W_rec", "B"]: # No layers to access

			W = W.flatten()

			W = W[W!=0]

			plt.hist(W, bins=(32 if wtype == "bias" else 256))
			plt.title(f"{wtype} (net {s})")
			plt.savefig(f"../vis/wdist/{wtype}_N{s}_dist.pdf")
			plt.close()

		else:
			for r in range(num_layers):
				W = W[s, r, :, :].flatten()
				W = W[W!=0]

				plt.hist(W, bins=256)
				plt.title(f"{wtype} (net {s}, layer {r})")
				plt.savefig(f"../vis/wdist/{wtype}_N{s}_R{r}_dist.pdf")
				plt.close()
