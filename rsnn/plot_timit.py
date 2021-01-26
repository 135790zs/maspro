import numpy as np
from config2 import cfg

inps = np.load(f"{cfg['wavs_fname']}_train_TIMIT.npy")
tars = np.load(f"{cfg['tars_fname']}_train_TIMIT.npy")

rng = np.random.default_rng()

inp = rng.choice
