from config import config
import numpy as np


def vtilde(v, z):
    return v - (v + 65) * z


def utilde(u, z):
    return u + 2 * z


def vnext(v, u, z, I):
    vtil = vtilde(v=v, z=z)
    return (vtil
            + config["dt"]*((0.04*vtil**2)
                            + 5*vtil
                            + 140
                            - utilde(u=u, z=z)
                            + I))


def unext(u, v, z):
    util = utilde(u=u, z=z)
    return (util
            + config["dt"]*(0.004 * vtilde(v=v, z=z)
                            - 0.02 * util))


def h(v):
    return config["gamma"] * np.exp((min(v, config["H1"]) - config["H1"])
                                    / config["H1"])


def evvnext(zi, zj, vi, vj, evv, evu):
    # term1 = (1 - zj)*(1 + (config["EVV1"] * vj + config["EVV2"]) * config["dt"]) \
    #          * evv
    term1 = (1 - zj
             + 0.08*config["dt"]*vj
             - 0.08*config["dt"]*zj*vj
             + 5*config["dt"]
             - 5*config["dt"]*zj) * evv
    term2 = - config["dt"] * evu
    term3 = zi * config["dt"]
    return term1 + term2 + term3


def evunext(zi, zj, evv, evu):
    term1 = 0.004 * config["dt"] * (1 - zj) * evv
    term2 = (1 - 0.02 * config["dt"]) * evu
    return term1 + term2


# TODO NEXT: Implement from Bellec directly, see if Traub is wrong or my own fns
# If latter: thoroughly check everything
