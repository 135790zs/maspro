from config import config as cfg
import numpy as np


def U_tilde(Nu, Nz):
    return Nu + cfg["refr1"] * Nz


def V_tilde(Nv, Nz):
    return Nv - (Nv - cfg["eqb"]) * Nz


def V_next(Nv, Nu, Nz, I):
    Nvt = V_tilde(Nv=Nv, Nz=Nz)
    Nut = U_tilde(Nu=Nu, Nz=Nz)

    return (Nvt + cfg["dt"] * (cfg["volt1"] * Nvt**2
                               + cfg["volt2"] * Nvt
                               + cfg["volt3"]
                               - Nut
                               + I))


def U_next(Nu, Nz, Nv):
    Nvt = V_tilde(Nv=Nv, Nz=Nz)
    Nut = U_tilde(Nu=Nu, Nz=Nz)

    return (Nut + cfg["dt"] * (cfg["refr2"] * Nvt
                               - cfg["refr3"] * Nut))


def EVv_next(EVv, EVu, Nz, Nv):
    return (EVv * (1 - Nz
                   + 2 * cfg["volt1"] * cfg["dt"] * Nv
                   - 2 * cfg["volt1"] * cfg["dt"] * Nv * Nz  # not sure about Nvp here
                   + cfg["volt2"] * cfg["dt"]
                   - cfg["volt2"] * cfg["dt"] * Nz)
            - EVu * cfg["dt"]
            + Nz[np.newaxis].T * cfg["dt"])


def EVu_next(EVv, EVu, Nz):
    return (EVv * (cfg["refr2"] * cfg["dt"]
                   - cfg["refr2"] * cfg["dt"] * Nz)
            + EVu * (1
                     - cfg["refr3"] * cfg["dt"]))


def H_next(Nv):
    return cfg["gamma"] * np.exp((np.clip(Nv, a_min=None, a_max=cfg["H1"]) - cfg["H1"])
                                 / cfg["H1"])
