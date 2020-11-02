import numpy as np
from config import cfg
from task import sinusoid, pulse, pulseclass


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

    for neuronvar in ["V", "Z", "Zbar", "I", "H"]:
        M[neuronvar] = np.zeros(shape=neuron_shape)

    for weightvar in ["ETbar", "EVV", "EVU", "DW", "ET"]:
        M[weightvar] = np.zeros(shape=weight_shape)

    M["T"] = np.zeros(shape=(cfg["Epochs"],))
    M["U"] = np.ones(shape=neuron_shape) * cfg["thr"]
    M["TZ"] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"])) * -cfg["dt_refr"]
    M["DW_out"] = np.zeros(shape=(cfg["Epochs"], cfg["N_R"],))
    M["B"] = rng.random(size=feedback_shape)
    M["W"] = rng.random(size=weight_shape)
    M["W_out"] = rng.random(size=(cfg["Epochs"], cfg["N_R"],))
    M["b_out"] = np.zeros(shape=(cfg["Epochs"], 1,))
    M['Y'] = np.zeros(shape=(cfg["Epochs"], cfg["N_O"],))
    M['error'] = np.zeros(shape=(cfg["Epochs"], cfg["N_O"],))
    M['loss'] = np.zeros(shape=(cfg["Epochs"],))
    M["Z_in"] = np.zeros(shape=(cfg["Epochs"],
                                cfg["N_Rec"],
                                cfg["N_R"]*2,))

    M["is_ALIF"] = np.zeros(shape=(cfg["N_Rec"] * cfg["N_R"]))
    M["is_ALIF"][:int(M["is_ALIF"].size * cfg["fraction_ALIF"])] = 1
    np.random.shuffle(M["is_ALIF"])
    M["is_ALIF"] = M["is_ALIF"].reshape((cfg["N_Rec"], cfg["N_R"]))

    if cfg["task"] == "narma10":
        M["X"] = rng.random(size=(cfg["Epochs"], cfg["N_I"])) * 0.5
    elif cfg["task"] == "sinusoid":
        M["X"] = sinusoid()
    elif cfg["task"] == "pulse":
        M["X"] = pulse()
    elif cfg["task"] == "pulseclass":
        M["X"] = pulseclass()["inp"]
        M['T'] = pulseclass()["tar"]

    M["XZ"] = rng.binomial(n=1, p=M["X"])

    for r in range(cfg["N_Rec"]):
        # Zero diag E: no self-conn
        np.fill_diagonal(M['W'][0, r, :, cfg["N_R"]:], 0)

    # M['W'][0, 0, 0, 0] = 20  # input 1 to neuron 1
    # M['W'][0, 0, 1, 0] = 0  # input 1 to neuron 2
    # M['W'][0, 0, 0, 1] = 0  # input 2 to neuron 1
    # M['W'][0, 0, 1, 1] = 20  # input 2 to neuron 2
    # # M['W'][0, 0, 1, 1] = 70  # input 2 to neuron 2
    # M['W'][0, 0, 0, 3] = 0  # n1 to n2
    # M['W'][0, 0, 1, 2] = 0  # n2 to n1

    return M


def temporal_filter(c, a):
    if a.shape[0] == 1:
        return a[0]
    return c * temporal_filter(c, a=a[:-1]) + a[-1]


def normalize(arr):
    return np.interp(arr, (arr.min(), arr.max()), (-1, 1))


def eprop_Z(t, TZ, V, U):
    return np.where(np.logical_and(t - TZ >= cfg["dt_refr"],
                                   V >= U),
                    1,
                    0)


def eprop_V(V, U, I, Z):
    return cfg["alpha"] * V + I - Z * cfg["thr"]


def eprop_U(V, U, Z, is_ALIF):
    return np.where(is_ALIF,
                    cfg["rho"] * U + Z,
                    U)


def eprop_EVV(EVV, Z_in):
    return cfg["alpha"] * EVV + Z_in


def eprop_EVU(H, EVV, EVU):
    return (H * EVV.T).T + ((cfg["rho"] - H * cfg["beta"]) * EVU.T).T


def eprop_H(t, V, U, is_ALIF):
    return 1 / cfg["thr"] * \
        cfg["gamma"] * np.clip(a=1 - (abs(V
                                          - cfg["thr"]
                                          - U * np.where(is_ALIF,
                                                         cfg["beta"],
                                                         1))
                                      / cfg["thr"]),
                               a_min=0,
                               a_max=None)


def eprop_ET(H, EVV, EVU, is_ALIF):
    return np.dot(H, EVV - cfg["beta"] * EVU)
