import numpy as np
from config import cfg


def task1(io_type, t):
    """ After every N inputs, system must spike once. N_I = N_O = 1."""

    interval = 100
    duration = 10
    strength_in = .5
    strength_out = 1.
    assert cfg["N_I"] == 1
    assert cfg["N_O"] == 1

    if io_type == "I":
        if t % interval < duration:
            return np.asarray([strength_in])
        else:
            return np.asarray([0.])
    elif io_type == "O":
        if t % interval == duration:
            return np.asarray([strength_out])
        else:
            return np.asarray([0.])
