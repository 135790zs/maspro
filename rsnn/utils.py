import numpy as np
from config import cfg
from task import sinusoid, pulse, pulseclass


def update_DWs(cfg, DW, err, M):

    # print(f"\t\tError: {M['error'][t]}")

    # M['DW'][t] = M['DW'][t-1] - cfg["eta"] * np.sum(
    #     W['B'][e-1] * M["error"][t]) * ut.temporal_filter(
    #         cfg["kappa"], M['ET'][:t+1])

    # # freeze input weights
    # M['DW'][t, 0, :, :cfg["N_R"]].fill(0.)
    # M["DW_out"][t] = -cfg["eta"] * np.sum(
    #     M["error"][t]) * ut.temporal_filter(
    #         cfg["kappa"], M['Z'][:t+1, -1])

    # M["Db_out"][t] = -cfg["eta"] * np.sum(M["error"][t])
    n_steps = M['XZ'].shape[0]
    for t in range(n_steps):  # maybe until 1 shorter
        B = DW['DW_out'].T  # TODO: NOT SURE!
        DW['DW'] = (DW['DW'] - cfg["eta"]
                    * np.sum(B * err[t]) * temporal_filter(
                        cfg["kappa"], M['ET'][:t+1]))
        # print(DW['DW'].shape, B.shape, err[t].shape)
    print(DW['DW_out'].shape,)
    DW['Db_out'] = -cfg["eta"] * np.sum(err, axis=0)
    DW['DW_out'] = (-cfg["eta"]
                    * np.outer(np.sum(err, axis=0),
                               temporal_filter(cfg["kappa"], M['Z'][:, -1])))

    # # Update weights
    # print(f"\tUpdating weights...")
    # if cfg["update_dead_weights"]:
    #     for r2 in range(cfg["N_Rec"]):
    #         # Zero diag E: no self-conn
    #         for t in range(cfg["Repeats"]):
    #             np.fill_diagonal(M['DW'][t, r2, :, cfg["N_R"]:], 0)
    # else:
    #     # Don't update zero-weights
    #     M['DW'] = np.where(W['W'], M['DW'], 0.)

    # W["B"][e] = W["B"][e-1] + np.sum(np.transpose(M["DW_out"], axes=(0, 2, 1)), axis=0)
    # W["B"][e] *= cfg["weight_decay"]
    # W['W'][e] = W['W'][e-1] + np.sum(M['DW'], axis=0)
    # W["W_out"][e] = W['W_out'][t] + np.sum(M['DW_out'], axis=0)
    # W["W_out"][e] *= cfg["weight_decay"]

    # W["b_out"][e] = W["b_out"][e-1] * np.sum(M["Db_out"], axis=0)
    return DW


def update_weight(cfg, DW, W):
    W['W'] += DW['DW']
    W['W'] *= cfg["weight_decay"]
    W['W_out'] += DW['DW_out']
    W['W_out'] *= cfg["weight_decay"]
    W['b_out'] += DW['Db_out']
    W['b_out'] *= cfg["weight_decay"]
    return W
    # M['DW'][t] = M['DW'][t-1] - cfg["eta"] * np.sum(
    #     W['B'][e-1] * M["error"][t]) * ut.temporal_filter(
    #         cfg["kappa"], M['ET'][:t+1])

    # # freeze input weights
    # M['DW'][t, 0, :, :cfg["N_R"]].fill(0.)
    # M["DW_out"][t] = -cfg["eta"] * np.sum(
    #     M["error"][t]) * ut.temporal_filter(
    #         cfg["kappa"], M['Z'][:t+1, -1])

    # M["Db_out"][t] = -cfg["eta"] * np.sum(M["error"][t])
    pass


def initialize_log():  # Vars for everything wiped after an example of N ms
    # rng = np.random.default_rng()
    M = {}
    neuron_shape = (cfg["Repeats"]*cfg["batch_size"],
                    cfg["N_Rec"],
                    cfg["N_R"],)
    weight_shape = (cfg["Repeats"]*cfg["batch_size"],
                    cfg["N_Rec"],
                    cfg["N_R"],
                    cfg["N_R"] * 2,)
    # feedback_shape = (cfg["Repeats"]*cfg["maxlen"],
    #                   cfg["N_Rec"],
    #                   cfg["N_R"],)

    for neuronvar in ["V", "Z", "Zbar", "I", "H"]:
        M[neuronvar] = np.zeros(shape=neuron_shape)

    for weightvar in ["ETbar", "EVV", "EVU", "DW", "ET"]:
        M[weightvar] = np.zeros(shape=weight_shape)

    M["T"] = np.zeros(shape=(cfg["Repeats"]*cfg["batch_size"],))
    M["U"] = np.ones(shape=neuron_shape) * cfg["thr"]
    M["TZ"] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"])) * -cfg["dt_refr"]
    M["DW_out"] = np.zeros(shape=(cfg["Repeats"]*cfg["batch_size"], cfg["N_O"], cfg["N_R"], ))
    M["Db_out"] = np.zeros(shape=(cfg["Repeats"]*cfg["batch_size"], cfg["N_O"],))
    # M["B"] = rng.random(size=feedback_shape)
    # M["W"] = rng.random(size=weight_shape)
    # M["W_out"] = rng.random(size=(cfg["Repeats"], cfg["N_R"], cfg["N_O"],))
    # M["b_out"] = np.zeros(shape=(cfg["Repeats"], cfg["N_O"],))
    M['Y'] = np.zeros(shape=(cfg["Repeats"]*cfg["batch_size"], cfg["N_O"],))
    M['error'] = np.zeros(shape=(cfg["Repeats"]*cfg["batch_size"], cfg["N_O"],))
    M['loss'] = np.zeros(shape=(cfg["Repeats"]*cfg["batch_size"],))
    M["Z_in"] = np.zeros(shape=(cfg["Repeats"]*cfg["batch_size"],
                                cfg["N_Rec"],
                                cfg["N_R"]*2,))

    M["is_ALIF"] = np.zeros(shape=(cfg["N_Rec"] * cfg["N_R"]))
    M["is_ALIF"][:int(M["is_ALIF"].size * cfg["fraction_ALIF"])] = 1
    np.random.shuffle(M["is_ALIF"])
    M["is_ALIF"] = M["is_ALIF"].reshape((cfg["N_Rec"], cfg["N_R"]))

    # for r in range(cfg["N_Rec"]):
    #     # Zero diag E: no self-conn
    #     np.fill_diagonal(M['W'][0, r, :, cfg["N_R"]:], 0)
    return M


def initialize_model(length):
    M = {}
    neuron_shape = (length,
                    cfg["N_Rec"],
                    cfg["N_R"],)
    weight_shape = (length,
                    cfg["N_Rec"],
                    cfg["N_R"],
                    cfg["N_R"] * 2,)

    for neuronvar in ["V", "Z", "Zbar", "I", "H"]:
        M[neuronvar] = np.zeros(shape=neuron_shape)

    for weightvar in ["ETbar", "EVV", "EVU", "DW", "ET"]:
        M[weightvar] = np.zeros(shape=weight_shape)

    M["T"] = np.zeros(shape=(length,))
    M["U"] = np.ones(shape=neuron_shape) * cfg["thr"]
    M["TZ"] = np.ones(shape=(cfg["N_Rec"], cfg["N_R"])) * -cfg["dt_refr"]

    M['error'] = np.zeros(shape=(length, cfg["N_O"],))
    M['loss'] = np.zeros(shape=(length,))
    M["Z_in"] = np.zeros(shape=(length, cfg["N_Rec"], cfg["N_R"]*2,))

    M["is_ALIF"] = np.zeros(shape=(cfg["N_Rec"] * cfg["N_R"]))
    M["is_ALIF"][:int(M["is_ALIF"].size * cfg["fraction_ALIF"])] = 1
    np.random.shuffle(M["is_ALIF"])
    M["is_ALIF"] = M["is_ALIF"].reshape((cfg["N_Rec"], cfg["N_R"]))

    return M


def initialize_weights():
    rng = np.random.default_rng()
    W = {}

    W["B"] = rng.random(
        size=(cfg["Epochs"], cfg["N_Rec"], cfg["N_R"], cfg["N_O"],))
    W["W"] = rng.random(
        size=(cfg["Epochs"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,))
    W["W_out"] = rng.random(size=(cfg["Epochs"], cfg["N_O"], cfg["N_R"],))
    W["b_out"] = np.zeros(shape=(cfg["Epochs"], cfg["N_O"],))

    return W


def get_error(M, tars, W_out, b_out, cfg):
    n_steps = M['XZ'].shape[0]
    error = np.zeros(shape=(n_steps, cfg["N_O"],))
    Y = np.zeros(shape=(n_steps, cfg["N_O"],))
    # Calculate network output
    for t in range(1, n_steps):
        Y[t] = (cfg["kappa"] * Y[t-1]
                + np.sum(W_out * M['Z'][t, -1])
                + b_out)
        error[t] = Y[t] - tars[t]
    return error


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
