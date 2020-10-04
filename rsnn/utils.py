from config import config
import numpy as np


def vtilde(v, z):
    return v - (v + config["IzhV1"]) * z


def utilde(u, z):
    return u + config["IzhU1"] * z


def vnext(v, u, z, I):
    vtil = vtilde(v=v, z=z)
    return (vtil
            + config["dt"]*((config["IzhV2"]*(vtil**2))
                            + config["IzhV3"]*vtil
                            + config["IzhV4"]
                            - utilde(u=u, z=z)
                            + I))


def unext(u, v, z):
    util = utilde(u=u, z=z)
    return (util
            + config["dt"]*(config["IzhU2"] * vtilde(v=v, z=z)
                            - config["IzhU3"] * util))


def h(v):
    return config["gamma"] * np.exp((min(v, config["H1"]) - config["H1"])
                                    / config["H1"])


def evvnext(zi, zj, vi, vj, evv, evu):
    # term1 = (1 - zj)*(1 + (config["EVV1"] * vj + config["EVV2"]) * config["dt"]) \
    #          * evv
    term1 = (1 - zj
             + config["EVV1"]*config["dt"]*vj
             - config["EVV1"]*config["dt"]*zj
             + 5*config["dt"]
             - 5*config["dt"]*zj) * evv
    term2 = (1 - config["EVU2"] * config["dt"]) * evu  # Substituting evu gets rid of exp, but leaves half
    term3 = zi * config["dt"]
    return sum([term1, term2, term3])


def evunext(zi, zj, evv, evu):
    term2 = - config["dt"] * evv
    term1 = config["EVU1"] * config["dt"] * (1 - zj) * evu
    return sum([term1, term2])


# TODO NEXT: Implement from Bellec directly, see if Traub is wrong or my own fns
# If latter: thoroughly check everything
