import os
import numpy as np
from config import cfg


def initialize_model(length, tar_size):
    M = {}
    neuron_shape = (length,
                    cfg["N_Rec"],
                    cfg["N_R"],)
    weight_shape = (length,
                    cfg["N_Rec"],
                    cfg["N_R"],
                    cfg["N_R"] * 2,)

    for neuronvar in ["V", "Z", "ZbarK", "I", "H"]:
        M[neuronvar] = np.zeros(shape=neuron_shape)

    for weightvar in ["EVV", "EVU", "ET", "DW", "ETbar"]:
        M[weightvar] = np.zeros(shape=weight_shape)

    M["U"] = np.ones(shape=neuron_shape) * cfg["thr"]
    M["TZ"] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"])) * -cfg["dt_refr"]

    M["Z_in"] = np.zeros(shape=(length, cfg["N_Rec"], cfg["N_R"] * 2,))
    M["Z_inbar"] = np.zeros(shape=(length, cfg["N_Rec"], cfg["N_R"] * 2,))

    M["DW_out"] = np.zeros(shape=(length, tar_size, cfg["N_R"],))
    M["DB"] = np.zeros(shape=(length, cfg["N_Rec"], cfg["N_R"], tar_size))
    M["Db_out"] = np.zeros(shape=(length, tar_size,))

    M["T"] = np.zeros(shape=(length, tar_size,))
    M["Y"] = np.zeros(shape=(length, tar_size,))
    M["P"] = np.zeros(shape=(length, tar_size,))
    M["Pmax"] = np.zeros(shape=(length, tar_size,))
    M["CE"] = np.zeros(shape=(length,))

    M["is_ALIF"] = np.zeros(shape=(cfg["N_Rec"] * cfg["N_R"]))
    M["is_ALIF"][:int(M["is_ALIF"].size * cfg["fraction_ALIF"])] = 1
    np.random.shuffle(M["is_ALIF"])
    M["is_ALIF"] = M["is_ALIF"].reshape((cfg["N_Rec"], cfg["N_R"]))

    return M


def initialize_weights(tar_size):
    rng = np.random.default_rng()
    W = {}

    W["B"] = rng.random(
        size=(cfg["Epochs"], cfg["N_Rec"], cfg["N_R"], tar_size,))
    W["W"] = rng.random(
        size=(cfg["Epochs"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,))
    W["W_out"] = rng.random(size=(cfg["Epochs"], tar_size, cfg["N_R"],))
    W["b_out"] = np.zeros(shape=(cfg["Epochs"], tar_size,))


    for r in range(cfg["N_Rec"]):
        # Zero diag recurrent W: no self-conn
        np.fill_diagonal(W['W'][0, r, :, cfg["N_R"]:], 0)

    W['W'][0, 0, 0, 0] = 3  # Input 1 to rec 1: frozen
    W['W'][0, 0, 1, 0] = 0  # Input 1 to rec 2
    W['W'][0, 0, 0, 1] = 0  # Input 2 to rec 1
    W['W'][0, 0, 1, 1] = 3  # Input 2 to rec 2: frozen
    W['W'][0, 0, 1, 2] = 1  # Rec 1 to rec 2
    W['W'][0, 0, 0, 3] = 1  # Rec 2 to rec 1
    # print(W['W'][0])


    return W


def get_error(M, tars, W_out, b_out):
    n_steps = M['X'].shape[0]
    error = np.zeros(shape=(n_steps, cfg["N_O"],))
    Y = np.zeros(shape=(n_steps, cfg["N_O"],))
    # Calculate network output
    for t in range(1, n_steps):
        Y[t] = (cfg["kappa"] * Y[t-1]
                + np.sum(W_out * M['Z'][t, -1])
                + b_out)
        error[t] = Y[t] - tars[t]
    return error


def save_weights(W, epoch):
    for k, v in W.items():
        np.save(f"{cfg['weights_fname']}/{k}", v[epoch])


def load_weights():
    W = {}
    for subdir, _, files in os.walk(cfg['weights_fname']):
        for filename in files:
            filepath = subdir + os.sep + filename
            W[filename[:-4]] = np.load(filepath)  # cut off '.npy'
    return W


def temporal_filter(c, a, depth=0):
    if depth == 32:
        return a[-1]
    if a.shape[0] == 1:
        return a[0]
    return c * temporal_filter(c, a=a[:-1], depth=depth+1) + a[-1:]


def normalize(arr):
    return np.interp(arr, (arr.min(), arr.max()), (-1, 1))


def eprop_Z(t, TZ, V, U):
    return np.where(np.logical_and(t - TZ >= cfg["dt_refr"],
                                   V >= U),
                    1,
                    0)


def eprop_V(V, I, Z):
    return cfg["alpha"] * V + I - Z * cfg["thr"]


def eprop_U(U, Z, is_ALIF):
    return np.where(is_ALIF,
                    cfg["rho"] * U + Z,
                    U)


def eprop_EVV(EVV, Z_in):
    return cfg["alpha"] * EVV + Z_in


def eprop_EVU(H, Z_inbar, EVU):
    return np.outer(H, Z_inbar) + (
        (cfg["rho"] - H[:, np.newaxis] * cfg["beta"]) * EVU)


def eprop_H(V, U, is_ALIF):
    return 1 / cfg["thr"] * \
        cfg["gamma"] * np.clip(a=1 - (abs(V
                                          - cfg["thr"]
                                          - U * np.where(is_ALIF,
                                                         cfg["beta"],
                                                         1))
                                      / cfg["thr"]),
                               a_min=0,
                               a_max=None)


def eprop_ET(H, EVV, EVU):
    return np.dot(H, EVV - cfg["beta"] * EVU)
