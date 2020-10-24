import numpy as np
from config import cfg


# def task1(io_type, t):
#     """ After every N inputs, system must spike once. N_I = N_O = 1."""

#     interval = 10
#     duration = 1
#     strength_in = 1
#     strength_out = 1.

#     if io_type == "I":
#         if t % interval < duration:
#             return np.asarray([strength_in] * cfg["N_I"])
#         return np.asarray([0.] * cfg["N_O"])

#     if t % interval == duration:
#         return np.asarray([strength_out] * cfg["N_I"])
#     return np.asarray([0.] * cfg["N_O"])


def narma10(t, u, y):
    y = np.append(np.zeros(shape=(10,)), y)
    u = np.append(np.zeros(shape=(10,)), u)
    new_y = (0.3 * y[-1]
             + 0.05 * y[-1] * np.sum(y[-10:])
             + 1.5 * u[t-10] * u[t-1]
             + 0.1)
    return new_y


def sinusoid(A=0.5, B=0.5, f=2.):
    return np.sin(np.arange(cfg["Epochs"]) / f) * A + B


def pulse(A=0.9, duration=20, gap=50):
    ret = np.zeros(shape=(cfg["Epochs"],))
    rng = np.random.default_rng()
    for t in range(cfg["Epochs"]):
        ret[t] = A * (t % (duration + gap) < duration)
    return ret
