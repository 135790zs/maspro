import numpy as np
from config import cfg


def narma10(t, u, y):
    y = np.append(np.zeros(shape=(10,)), y)
    u = np.append(np.zeros(shape=(10,)), u)
    new_y = (0.3 * y[-1]
             + 0.05 * y[-1] * np.sum(y[-10:])
             + 1.5 * u[t-10] * u[t-1]
             + 0.1)
    return np.expand_dims(new_y, axis=1)


def pulse(A=(1, 1, 1), duration=(1, 1, 10), gap=(20, 20, 40), offset=(0, 6, 2)):
    A = A[:cfg["N_I"]]
    gap = gap[:cfg["N_I"]]
    duration = duration[:cfg["N_I"]]
    offset = offset[:cfg["N_I"]]

    ret = np.zeros(shape=(cfg["Steps"], cfg["N_I"]))

    for t in range(cfg["Steps"]):
        for n in range(cfg["N_I"]):
            ret[t, n] = A[n] * ((t - offset[n]) % (duration[n] + gap[n]) < duration[n])
    return ret


def sinusoid(A=(0.3, 0.3, 0.5), f=(4, 1, 11), phase=(0, 0, 0)):
    A = A[:cfg["N_I"]]
    f = f[:cfg["N_I"]]
    phase = phase[:cfg["N_I"]]
    ret = np.zeros(shape=(cfg["Steps"], cfg["N_I"]))

    for n in range(cfg["N_I"]):
        for t in range(cfg["Steps"]):
            ret[:, n] = np.sin((np.arange(cfg["Steps"]) + phase[n]) / f[n]) * A[n] + A[n]
    return ret


def pulseclass(A=(1, 1, 1,), duration=(3, 3, 3,), gap=(5, 5, 5,), offset=(0, 2, 4)):
    A = A[:cfg["N_I"]]
    gap = gap[:cfg["N_I"]]
    duration = duration[:cfg["N_I"]]
    offset = offset[:cfg["N_I"]]

    inp = np.zeros(shape=(cfg["Steps"], cfg["N_I"]))
    tar = np.zeros(shape=(cfg["Steps"], cfg["N_O"]))

    for t in range(cfg["Steps"]):
        for n in range(cfg["N_I"]):
            inp[t, n] = A[n] * ((t - offset[n]) % (duration[n] + gap[n]) < duration[n])
    for t in range(cfg["Steps"]):  # classidx is sum of active
        for n in range(cfg["N_O"]):
            ix = int(np.sum(inp[t, :]))
            tar[t, ix] = 1

    return {"inp": inp, "tar": tar}