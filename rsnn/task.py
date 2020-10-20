import numpy as np
from config import cfg


def task1(io_type, t):
    """ After every N inputs, system must spike once. N_I = N_O = 1."""

    interval = 10
    duration = 1
    strength_in = 1
    strength_out = 1.

    if io_type == "I":
        if t % interval < duration:
            return np.asarray([strength_in] * cfg["N_I"])
        return np.asarray([0.] * cfg["N_O"])

    if t % interval == duration:
        return np.asarray([strength_out] * cfg["N_I"])
    return np.asarray([0.] * cfg["N_O"])


def narma10(t, u, y):
    y = np.append(np.zeros(shape=(10,)), y)
    u = np.append(np.zeros(shape=(10,)), u)
    new_y = (0.3 * y[-1]
             + 0.05 * y[-1] * np.sum(y[-10:])
             + 1.5 * u[t-10] * u[t-1]
             + 0.1)
    return new_y
