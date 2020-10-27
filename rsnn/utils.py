import numpy as np
from config import cfg
from task import sinusoid, pulse


def initialize_log():
    rng = np.random.default_rng()
    M = {}
    neuron_shape = (cfg["Epochs"],
                    cfg["N_Rec"],
                    cfg["N_R"],)
    weight_shape = (cfg["Epochs"],
                    cfg["N_Rec"],
                    cfg["N_R"],
                    cfg["N_R"] * 2,)
    feedback_shape = (cfg["Epochs"],
                      cfg["N_Rec"],
                      cfg["N_R"],)

    if cfg["neuron"] == "Izhikevich":
        M["V"] = np.ones(shape=neuron_shape) * cfg["eqb"]
    else:
        M["V"] = np.zeros(shape=neuron_shape)

    M["Z"] = np.zeros(shape=neuron_shape)
    M["I"] = np.zeros(shape=neuron_shape)
    M["H"] = np.zeros(shape=neuron_shape)
    if cfg["neuron"] == "ALIF":
        M["U"] = np.ones(shape=neuron_shape) * cfg["thr"]
    else:
        M["U"] = np.zeros(shape=neuron_shape)
    M["TZ"] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"])) * -cfg["dt_refr"]
    M["EVV"] = np.zeros(shape=weight_shape)
    M["EVU"] = np.zeros(shape=weight_shape)
    M["DW"] = np.zeros(shape=weight_shape)
    M["ET"] = np.zeros(shape=weight_shape)
    M["W"] = rng.random(size=weight_shape) * 10
    M["B"] = np.ones(shape=feedback_shape) * rng.random()
    M["B"] = rng.random(size=feedback_shape)

    M["W_out"] = rng.random(size=(cfg["N_R"],))
    M["b_out"] = 0  # rng.random()

    M['Y'] = np.zeros(shape=(cfg["Epochs"],))
    M['error'] = np.zeros(shape=(cfg["Epochs"],))

    if cfg["task"] == "narma10":
        M["X"] = rng.random(size=(cfg["Epochs"],)) * 0.5
    elif cfg["task"] == "sinusoid":
        M["X"] = sinusoid()
    elif cfg["task"] == "pulse":
        M["X"] = pulse()

    M["T"] = M["X"][:, 0]

    M["XZ"] = rng.binomial(n=1, p=M["X"])

    for r in range(cfg["N_Rec"]):
        # Zero diag E: no self-conn
        np.fill_diagonal(M['W'][0, r, :, cfg["N_R"]:], 0)
    # M['W'][0, 0, 1, 0] = 0  # input 1 to neuron 2
    # M['W'][0, 0, 0, 1] = 0  # input 2 to neuron 1
    # M['W'][0, 0, 0, 0] = 70  # input 1 to neuron 1
    # M['W'][0, 0, 1, 1] = 70  # input 2 to neuron 2
    # M['W'][0, 0, 0, 3] = 1  # n1 to n2
    # M['W'][0, 0, 1, 2] = 1  # n2 to n1

    return M


def rep_along_axis(arr):
    return np.repeat(arr[np.newaxis], cfg["N_R"]*2, axis=0).T


def EMA(arr, arr_ema, ep):
    return (cfg["EMA"] * arr[ep] + (1 - cfg["EMA"]) * arr_ema[ep-1]) \
            if ep else arr[ep]


def normalize(arr):
    return np.interp(arr, (arr.min(), arr.max()), (-1, 1))


def errfn(a1, a2):
    return np.sum(np.abs(a1 - a2), axis=1)


def eprop_Z(t, TZ, V, U):
    if cfg["neuron"] == "LIF":
        # Note: diff from Traub! 0 during whole refr
        return np.where(np.logical_and(t - TZ >= cfg["dt_refr"],
                                       V >= cfg["thr"]),
                        1,
                        0)
    elif cfg["neuron"] == "ALIF":
        return np.where(np.logical_and(t - TZ >= cfg["dt_refr"],
                                       V >= U),
                        1,
                        0)
    # Izhikevich
    return np.where(V >= cfg["thr"], 1, 0)


def Ut(U, Z):
    return U + cfg["refr1"] * Z


def Vt(V, Z):
    return V - (V - cfg["eqb"]) * Z


def eprop_V(V, U, I, Z, R):
    if cfg["neuron"] in ["LIF", "ALIF"]:
        return (cfg["alpha"] * V
                + I
                - Z * cfg["alpha"] * V
                - R * cfg["alpha"] * V)

    # Izhikevich
    Vt_ = Vt(V=V, Z=Z)
    return Vt_ + cfg["dt"] * (cfg["volt1"] * Vt_**2
                              + cfg["volt2"] * Vt_
                              + cfg["volt3"]
                              - Ut(U=U, Z=Z)
                              + I)


def eprop_U(V, U, Z):
    if cfg["neuron"] == "LIF":
        return U
    elif cfg["neuron"] == "ALIF":
        return cfg["rho"] * U + Z

    # Izhikevich
    Ut_ = Ut(U=U, Z=Z)
    return Ut_ + cfg["dt"] * (cfg["refr2"] * Vt(V=V, Z=Z)
                              - cfg["refr3"] * Ut_)


def eprop_EVV(EVV, EVU, Z, V, R, Z_in):
    if cfg["neuron"] in ["LIF", "ALIF"]:  # LIF has no U or EVU
        return (EVV * cfg["alpha"] * rep_along_axis(arr=(1 - Z - R))
                + Z_in)

    # Izhikevich
    return (rep_along_axis((1 - Z) * (1 + (2 * cfg["volt1"] * V
                           + cfg["volt2"]) * cfg["dt"])) * EVV
            - cfg["dt"] * EVU
            + cfg["dt"] * Z_in)


def eprop_EVU(H, Z, EVV, EVU):
    if cfg["neuron"] in ["LIF", "ALIF"]:
        H = rep_along_axis(arr=H)
        # return H * EVV + (cfg["rho"] - H * cfg["beta"]) * EVU
        return H * EVV + cfg["rho"] * EVU

    # Izhikevich
    return (cfg["refr2"] * cfg["dt"] * EVV * rep_along_axis(1 - Z)
            + EVU * (1 - cfg["refr3"] * cfg["dt"]))


def eprop_H(t, TZ, V, U):
    if cfg["neuron"] == "LIF":
        return np.where(t - TZ < cfg["dt_refr"],
                        -cfg["gamma"],
                        cfg["gamma"] * np.clip(a=1 - (abs(V - cfg["thr"])
                                                      / cfg["thr"]),
                                               a_min=0,
                                               a_max=None))
    elif cfg["neuron"] == "ALIF":
        return np.where(t - TZ < cfg["dt_refr"],
                        -cfg["gamma"],
                        cfg["gamma"] * np.clip(a=1 - (abs(V
                                                          - cfg["thr"]
                                                          - cfg["beta"] * U)
                                                      / cfg["thr"]),
                                               a_min=0,
                                               a_max=None))

    # Izhikevich
    return cfg["gamma"] * np.exp((np.clip(V,
                                          a_min=None,
                                          a_max=cfg["thr"]) - cfg["thr"])
                                 / cfg["thr"])


def eprop_ET(H, EVV, EVU):
    if cfg["neuron"] in ["LIF", "Izhikevich"]:
        return rep_along_axis(H) * EVV

    # ALIF
    return rep_along_axis(H) * (EVV - cfg["beta"] * EVU)
