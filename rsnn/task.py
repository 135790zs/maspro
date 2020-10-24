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


def sinusoid(A=0.5, B=0.5, f=2.):
    return np.sin(np.arange(cfg["Epochs"]) / f) * A + B


def pulse(A=0.9, duration=20, gap=50):
    ret = np.zeros(shape=(cfg["Epochs"],))
    for t in range(cfg["Epochs"]):
        ret[t] = A * (t % (duration + gap) < duration)
    return ret
