import numpy as np
from config import cfg


def narma10(t, u, y):
    y = np.append(np.zeros(shape=(10,)), y)
    u = np.append(np.zeros(shape=(10,)), u)
    new_y = (0.3 * y[-1]
             + 0.05 * y[-1] * np.sum(y[-10:])
             + 1.5 * u[t-10] * u[t-1]
             + 0.1)
    return new_y


def sinusoid(A=0.5, B=0.5, f=2):
    sig = np.sin(np.arange(cfg["Epochs"]) / f) * A + B
    ret = np.expand_dims(sig, axis=1)
    return ret


# def pulse(A=1, duration=20, gap=50):
#     ret = np.zeros(shape=(cfg["Epochs"],))
#     for t in range(cfg["Epochs"]):
#         ret[t] = A * (t % (duration + gap) < duration)
#     return ret


def pulse(A=(1, 1, 1), duration=(10, 10, 10), gap=(40, 40, 40), offset=(0, 1, 2)):
    A = A[:cfg["N_I"]]
    gap = gap[:cfg["N_I"]]
    duration = duration[:cfg["N_I"]]
    offset = offset[:cfg["N_I"]]

    ret = np.zeros(shape=(cfg["Epochs"], cfg["N_I"]))

    for t in range(cfg["Epochs"]):
        for n in range(cfg["N_I"]):
            ret[t, n] = A[n] * ((t - offset[n]) % (duration[n] + gap[n]) < duration[n])
    return ret
