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
    M["W"] = rng.random(size=weight_shape)
    M["B"] = np.ones(shape=feedback_shape) * rng.random()

    M["W_out"] = rng.random(size=(cfg["N_R"],))
    M["b_out"] = rng.random()

    M['Y'] = np.zeros(shape=(cfg["Epochs"],))
    M['error'] = np.zeros(shape=(cfg["Epochs"],))

    if cfg["task"] == "narma10":
        M["X"] = rng.random(size=(cfg["Epochs"],)) * 0.5
    elif cfg["task"] == "sinusoid":
        M["X"] = sinusoid()
    elif cfg["task"] == "pulse":
        M["X"] = pulse()

    M["T"] = M["X"]

    M["XZ"] = rng.binomial(n=1, p=M["X"])

    for r in range(cfg["N_Rec"]):
        np.fill_diagonal(M['W'][0, r, :, cfg["N_R"]:], 0)  # Zero diag E: no self-conn
    # M['W'][0, 0, 0, 0] = 50  # Custom set input to layer 0 neuron 0
    # M['W'][0, 1, 0, 0] = 0  # Custom set input to layer 1 neuron 0

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


def rep_along_axis(arr):
    return np.repeat(arr[np.newaxis], cfg["N_R"]*2, axis=0).T


def EMA(arr, arr_ema, ep):
    return (cfg["EMA"] * arr[ep] + (1 - cfg["EMA"]) * arr_ema[ep-1]) \
            if ep else arr[ep]


def eprop_UNUSED(model, M, X, t):
    # TODO: Weird what order of W, Z should be. Seems diff for units and
    # rsnn

    # MOVE DOT AFTER THR?

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

    Iz = np.dot(M['W'], M['Z'])
    Iz += X

    M['TZ'] = np.where(M['Z'], t, M['TZ'])

    if model in ["LIF", "ALIF"]:
        R = (t - M['TZ'] == cfg["dt_refr"]).astype(int)
        M['V'] = (cfg["alpha"] * M['V']
                  + Iz
                  - M['Z'] * cfg["alpha"] * M['V']
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
                                - 2 * cfg["volt1"] * cfg["dt"] * M['V'] * M['Z']
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


def normalize(arr):
    return np.interp(arr, (arr.min(), arr.max()), (-1, 1))


def errfn(a1, a2):
    return np.sum(np.abs(a1 - a2), axis=1)


def eprop_Z(t, TZ, V, U):
    if cfg["neuron"] == "LIF":
        return np.where(np.logical_and(t - TZ >= cfg["dt_refr"],  # Note: diff from Traub! 0 during whole refr
                                       V >= cfg["thr"]),
                        1,
                        0)
    elif cfg["neuron"] == "ALIF":
        return np.where(np.logical_and(t - TZ >= cfg["dt_refr"],  # Note: diff from Traub! 0 during whole refr
                                       V >= U),
                        1,
                        0)
    elif cfg["neuron"] == "Izhikevich":
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
    elif cfg["neuron"] == "Izhikevich":
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
    elif cfg["neuron"] == "Izhikevich":
        Ut_ = Ut(U=U, Z=Z)
        return Ut_ + cfg["dt"] * (cfg["refr2"] * Vt(V=V, Z=Z)
                                  - cfg["refr3"] * Ut_)


def eprop_EVV(EVV, EVU, Z, V, R, Z_in):
    if cfg["neuron"] in ["LIF", "ALIF"]:
        return (EVV * cfg["alpha"] * rep_along_axis(arr=(1 - Z - R))
                + Z_in)
    elif cfg["neuron"] == "Izhikevich":
        return (EVV * (1 - Z) * (1 + (2 * cfg["volt1"] * V
                                      + cfg["volt2"]) * cfg["dt"])
                - cfg["dt"] * EVU
                + cfg["dt"] * Z_in)


def eprop_EVU(H, Z, EVV, EVU, R):
    if cfg["neuron"] in ["LIF", "ALIF"]:
        # TEST = cfg["alpha"] * (1 - Z - R)
        # print(H.shape, TEST.shape)
        H = rep_along_axis(arr=H)
        return H * EVV + (cfg["rho"] - H * cfg["beta"]) * EVU
    elif cfg["neuron"] == "Izhikevich":
        return (cfg["refr2"] * cfg["dt"] * EVV * (1 - Z)
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
    elif cfg["neuron"] == "Izhikevich":
        return cfg["gamma"] * np.exp((np.clip(V,
                                              a_min=None,
                                              a_max=cfg["thr"]) - cfg["thr"])
                                     / cfg["thr"])


def eprop_ET(H, EVV, EVU):
    if cfg["neuron"] in ["LIF", "Izhikevich"]:
        return rep_along_axis(H) * EVV
    elif cfg["neuron"] == "ALIF":
        return rep_along_axis(H) * (EVV - cfg["beta"] * EVU)
