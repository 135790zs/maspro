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


def evvnext(zi, zj, vj, evv, evu):
    return ((1 + config["EVV1"] * vj + config["EVV2"])
            * evv * (1 - zj) * config["dt"]  # switch evu evv?
            - evu * config["dt"]
            + zi * config["dt"])


def evunext(zi, zj, evv, evu):
    return (config["EVU1"] * config["dt"] * (1 - zj) * evv
            + evu
            - evu * config["EVU2"] * config["dt"])
