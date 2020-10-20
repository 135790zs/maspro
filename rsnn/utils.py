import numpy as np
from config import cfg


def initialize_log():
    rng = np.random.default_rng()
    M = {}
    neuron_shape = (cfg["Epochs"],
                    cfg["N_Rec"],
                    cfg["N_R"],)
    weight_shape = (cfg["Epochs"],
                    cfg["N_Rec"] - 1,
                    cfg["N_R"] * 2,
                    cfg["N_R"] * 2,)
    feedback_shape = (cfg["Epochs"],
                      cfg["N_Rec"] - 2,
                      cfg["N_R"],)

    if cfg["neuron"] == "Izhikevich":
        M["V"] = np.ones(shape=neuron_shape) * cfg["eqb"]
    else:
        M["V"] = np.zeros(shape=neuron_shape)

    M["U"] = np.zeros(shape=neuron_shape)
    M["Z"] = np.zeros(shape=neuron_shape)
    M["TZ"] = np.ones(shape=neuron_shape) * -cfg["dt_refr"]
    M["H"] = np.zeros(shape=neuron_shape)
    M["EVV"] = np.zeros(shape=weight_shape)
    M["EVU"] = np.zeros(shape=weight_shape)
    M["ET"] = np.zeros(shape=weight_shape)
    M["W"] = rng.random(size=weight_shape) * 4 - 2
    M["B"] = np.ones(shape=feedback_shape) * rng.random()
    M["L"] = np.ones(shape=feedback_shape)
    M["input_spike"] = np.zeros(shape=(cfg["Epochs"], cfg["N_I"]))
    M["output"] = np.zeros(shape=(cfg["Epochs"], cfg["N_O"]))
    M["output_EMA"] = np.zeros(shape=(cfg["Epochs"], cfg["N_O"]))
    M["target"] = np.zeros(shape=(cfg["Epochs"], cfg["N_O"]))
    M["target_EMA"] = np.zeros(shape=(cfg["Epochs"], cfg["N_O"]))

    if cfg["task"] == "narma10":
        M["input"] = rng.random(size=(cfg["Epochs"], cfg["N_I"])) * 0.5
    else:
        M["input"] = np.zeros(shape=(cfg["Epochs"], cfg["N_I"]))

    for r in range(cfg["N_Rec"]-1):
        M['W'][0, r, :, :] = drop_weights(W=M['W'][0, r, :, :],
                                          layer=r)


    return M


def get_artificial_input(T, num, dur, diff, interval, val, switch_interval):
    X = np.zeros(shape=(T, num))
    for t in range(0, T):
        if t % (switch_interval*2) < switch_interval:
            X[t, 0] = val if t % interval <= dur else 0
            X[t, 1] = val if (t % interval <= diff + dur
                              and t % interval > diff) else 0
        else:
            X[t, 0] = val if (t % interval <= diff + dur
                              and t % interval > diff) else 0
            X[t, 1] = val if t % interval <= dur else 0
    return X


def EMA(arr, arr_ema, ep):
    return (cfg["EMA"] * arr[ep, :] + (1 - cfg["EMA"]) * arr_ema[ep, :]) \
            if ep else arr[ep, :]


def eprop(model, M, X, t, uses_weights):
    # TODO: Weird what order of W, Z should be. Seems diff for units and
    # rsnn
    Iz = (np.dot(M['Z'], M['W'], )
          if uses_weights else np.zeros(shape=M['Z'].shape))
    Iz += X

    if model == "LIF":
        M['Z'] = np.where(np.logical_and(t - M['TZ'] >= cfg["dt_refr"],
                                         M['V'] >= cfg["thr"]),
                          1,
                          0)
    elif model == "ALIF":
        M['Z'] = np.where(np.logical_and(t - M['TZ'] >= cfg["dt_refr"],
                                         M['V'] >= (cfg["thr"]
                                                    + cfg["beta"] * M['U'])),
                          1,
                          0)
    elif model == "Izhikevich":
        M['Z'] = np.where(M['V'] >= cfg["thr"], 1., 0.)

    M['TZ'] = np.where(M['Z'], t, M['TZ'])

    if model in ["LIF", "ALIF"]:
        R = (t - M['TZ'] == cfg["dt_refr"]).astype(int)

        M['V'] = (cfg["alpha"] * M['V']
                  + Iz - M['Z'] * cfg["alpha"] * M['V']
                  - R * cfg["alpha"] * M['V'])

    elif model == "Izhikevich":
        Vt = M['V'] - (M['V'] - cfg["eqb"]) * M['Z']
        Ut = M['U'] + cfg["refr1"] * M['Z']
        Vn = Vt + cfg["dt"] * (cfg["volt1"] * Vt**2
                               + cfg["volt2"] * Vt
                               + cfg["volt3"]
                               - Ut
                               + Iz)
        Un = Ut + cfg["dt"] * (cfg["refr2"] * Vt
                               - cfg["refr3"] * Ut)

    if model == "ALIF":
        M['U'] = cfg["rho"] * M['U'] + M['Z']

    if model in ["LIF", "ALIF"]:
        M['EVV'] = (cfg["alpha"] * (1 - M['Z'] - R) * M['EVV']
                    + M['Z'][np.newaxis].T)

    elif model == "Izhikevich":
        M['EVV'] = (M['EVV'] * (1 - M['Z']
                                + 2 * cfg["volt1"] * cfg["dt"] * M['V']
                                - 2 * cfg["volt1"] * cfg["dt"] * M['V']
                                * M['Z']
                                + cfg["volt2"] * cfg["dt"]
                                - cfg["volt2"] * cfg["dt"] * M['Z'])
                    - M['EVU']  # Traub: "* cfg["dt"]" may have to be appended
                    + M['Z'][np.newaxis].T * cfg["dt"])
        M['EVU'] = (cfg["refr2"] * cfg["dt"] * M['EVV'] * (1 - M['Z'])
                    + M['EVU'] * (1 - cfg["refr3"] * cfg["dt"]))

    if model == "LIF":
        M['H'] = np.where(t - M['TZ'] < cfg["dt_refr"],
                          -cfg["gamma"],
                          cfg["gamma"] * np.clip(a=1 - (abs(M['V']
                                                            - cfg["thr"])
                                                        / cfg["thr"]),
                                                 a_min=0,
                                                 a_max=None))
    elif model == "ALIF":
        M['H'] = np.where(t - M['TZ'] < cfg["dt_refr"],
                          -cfg["gamma"],
                          cfg["gamma"] * np.clip(
                              a=1 - (abs(M['V'] - (cfg["thr"]
                                                   + cfg["beta"] * M['U']))
                                     / cfg["thr"]),
                              a_min=0,
                              a_max=None))
        M['EVU'] = M['H'] * M['EVV'] + (cfg["rho"]
                                        - M['H'] * cfg["beta"]) * M['EVU']
    elif model == "Izhikevich":
        M['V'] = Vn
        M['U'] = Un
        M['H'] = cfg["gamma"] * np.exp((np.clip(M['V'],
                                                a_min=None,
                                                a_max=cfg["thr"]) - cfg["thr"])
                                       / cfg["thr"])

    if model in ["LIF", "Izhikevich"]:
        M['ET'] = M['H'] * M['EVV']
    elif model == "ALIF":
        M['ET'] = M['H'] * (M['EVV'] - cfg["beta"] * M['EVU'])

    if M['L'].shape[-1] == M['ET'].shape[-1]:
        M['ET'] = M['ET'] * M['L']
    else:
        M['ET'] = M['ET'] * np.repeat(a=M['L'], repeats=2)

    D = np.where(M['W'], M['ET'], 0)  # only update nonzero
    M['W'] = M['W'] + D

    return M


def drop_weights(W, layer):
    """ Done for initial epoch, zero weights will never be updated."""
    N = W.shape[0]//2

    W[:, N:] = 0  # Zero full E: can't go back, nor recur next layer

    if layer != 0:
        np.fill_diagonal(W[:N, :N], 0)  # Zero diag NW: no self-connections
    else:
        W[:N, :N] = 0  # empty NW: don't recur input layer
        W[N:, cfg["N_I"]:N] = 0  # No connections from nonexisting input

    if layer == cfg["N_R"] - 2:
        W[N+cfg["N_O"]:, :N] = 0  # No connections to nonexisting output

    print(W)
    return W


def normalize(arr):
    return np.interp(arr, (arr.min(), arr.max()), (-1, 1))


def errfn(a1, a2):
    return np.sum(np.abs(a1 - a2), axis=1)
