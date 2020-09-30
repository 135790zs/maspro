import numpy as np
from config import config


def random_input():
    rng = np.random.default_rng()

    return rng.integers(2, size=config["N_I"])
