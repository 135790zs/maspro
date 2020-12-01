import os
import numpy as np
import datetime
import json


def initialize_model(cfg, length, tar_size):
    """ Initializes the variables used in predicting a single time series.

        These are the variables used in e-prop, and the reason they're
        stored for every time step (rather than overwriting per time step)
        is so the evolution of the network can be inspected after running.

        V:       Neuron voltage
        U:       Neuron threshold adaptation factor
        Z:       Neuron outgoing spike train
        Zbar:    Low-pass filter of outgoing spike train
        TZ:      Neuron time of last outgoing spike
        Z_in:    Neuron incoming spike train (nonbinary for first layer)
        Z_inbar: Low-pass filter of incoming spike train
        TZ_in:   Neuron time of last incoming spike
        H:       Timestep pseudo-derivative
        I:       Weighted input to neuron
        EVV:     Synapse voltage eligibility
        EVU:     Synapse threshold adaptation eligibility
        ET:      Synapse eligibility trace
        gW:      Network weights gradient
        DW:      Network weights update
        DB:      Broadcast weights update
        dW_out:  Output weights update
        gW_out:  Output weights gradient
        db_out:  Bias update
        gb_out:  Bias gradient
        T:       Timestep target
        Y:       Timestep output
        P:       Timestep probability vector
        Pmax:    Timestep prediction
        CE:      Timestep cross-entropy loss
        L:       Timestep learning signal
        is_ALIF: Mask determining which neurons have adaptive thresholds
    """

    M = {}
    neuron_shape = (cfg["n_directions"],
                    length,
                    cfg["N_Rec"],
                    cfg["N_R"],)

    neuron_timeless_shape = (cfg["n_directions"],
                             cfg["N_Rec"],
                             cfg["N_R"],)

    weight_shape = (cfg["n_directions"],
                    length,
                    cfg["N_Rec"],
                    cfg["N_R"],
                    cfg["N_R"] * 2,)
    Z_in_shape = (cfg["n_directions"],
                  length,
                  cfg["N_Rec"],
                  cfg["N_R"] * 2,)

    W_out_shape = (cfg["n_directions"],
                   length,
                   tar_size,
                   cfg["N_R"],)

    b_out_shape = (cfg["n_directions"],
                   length,
                   tar_size,)

    T_shape = (length, tar_size,)

    for T_var in ["T", "P", "Pmax"]:
        M[T_var] = np.zeros(shape=T_shape)

    for neuronvar in ["V", "Z", "Zbar", "I", "H", "L"]:
        M[neuronvar] = np.zeros(shape=neuron_shape)

    for weightvar in ["EVV", "EVU", "ET", "DW", "ETbar", 'gW']:
        M[weightvar] = np.zeros(shape=weight_shape)

    for z_in_var in ["Z_in", "Z_inbar"]:
        M[z_in_var] = np.zeros(shape=Z_in_shape)

    for W_out_var in ["DW_out", "gW_out"]:
        M[W_out_var] = np.zeros(shape=W_out_shape)

    for b_out_var in ["Db_out", "gb_out"]:
        M[b_out_var] = np.zeros(shape=b_out_shape)

    M["U"] = np.ones(shape=neuron_shape) * cfg["thr"]

    M["TZ"] = np.ones(shape=neuron_timeless_shape) * -cfg["dt_refr"]

    M["TZ_in"] = np.zeros(shape=(cfg["n_directions"],
                                 cfg["N_Rec"],
                                 cfg["N_R"] * 2,))

    M["DB"] = np.zeros(shape=(cfg["n_directions"],
                              length,
                              cfg["N_Rec"],
                              cfg["N_R"],
                              tar_size))

    M["Y"] = np.zeros(shape=(cfg["n_directions"], length, tar_size,))
    M["CE"] = np.zeros(shape=(length,))

    M["is_ALIF"] = np.zeros(
        shape=(cfg["n_directions"] * cfg["N_Rec"] * cfg["N_R"]))

    M["is_ALIF"][:int(M["is_ALIF"].size * cfg["fraction_ALIF"])] = 1
    np.random.shuffle(M["is_ALIF"])

    M["is_ALIF"] = M["is_ALIF"].reshape(neuron_timeless_shape)

    return M


def initialize_weights(cfg, tar_size):
    """ Initializes the variables used to train a network's weights.

    The difference with the Model is that the model re-initializes for
    every new time series, while the Weights in this function are persistent
    over many epochs. They are also stored in an array such that the
    evolution of the weights over the number of epochs can be inspected.

    W_out: Output weights
    b_out: Bias
    W:     Network weights
    B:     Broadcast weights
    """

    W = {}
    rng = np.random.default_rng()

    W["W"] = rng.random(size=(cfg["Epochs"],
                              cfg["n_directions"],
                              cfg["N_Rec"],
                              cfg["N_R"],
                              cfg["N_R"] * 2,))

    # Decrease weights in first layer
    W["W"][0, :, 0] *= 2 / cfg["N_R"]

    W["W_out"] = rng.random(size=(cfg["Epochs"],
                                  cfg["n_directions"],
                                  tar_size,
                                  cfg["N_R"],))

    W["b_out"] = np.zeros(shape=(cfg["Epochs"],
                                 cfg["n_directions"],
                                 tar_size,))

    B_shape = (cfg["Epochs"],
               cfg["n_directions"],
               cfg["N_Rec"],
               cfg["N_R"],
               tar_size,)

    if cfg["eprop_type"] == "random":  # Gaussian, variance of 1
        W["B"] = rng.normal(size=B_shape, scale=1)

    elif cfg["eprop_type"] == "adaptive":  # Gaussian, variance of 1/N
        W["B"] = rng.normal(size=B_shape, scale=np.sqrt(1 / cfg["N_R"]))

    else:  # Uniform [0, 1]
        W["B"] = rng.random(size=B_shape)

    # Drop all self-looping weights. A neuron cannot be connected with itself.
    for r in range(cfg["N_Rec"]):
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W['W'][0, s, r, :, cfg["N_R"]:], 0)

    # W['W'][0, 0, 0, 0, 0] = 2  # Input 1 to rec 1: frozen
    # W['W'][0, 0, 0, 1, 0] = 0  # Input 1 to rec 2
    # W['W'][0, 0, 0, 0, 1] = 0  # Input 2 to rec 1
    # W['W'][0, 0, 0, 1, 1] = 0  # Input 2 to rec 2: frozen
    # W['W'][0, 0, 0, 1, 2] = 2  # Rec 1 to rec 2 (B)
    # W['W'][0, 0, 0, 0, 3] = 0  # Rec 2 to rec 1 (T)

    return W


def save_weights(W, epoch, log_id):
    """ Saves well-performing weights to files, to be loaded later.

    Every individual run gets its own weight checkpoints.
    """

    for k, v in W.items():
        # Every type of weight (W, W_out, b_out, B) gets its own file.
        np.save(f"../log/{log_id}/checkpoints/{k}", v[epoch])


def load_weights(log_id):
    """ Load weights from file to be used in network again.

    Mainly used for testing purposes, because the weights corresponding
    to the lowest validation loss are stored.
    """

    W = {}
    for subdir, _, files in os.walk(f"../log/{log_id}/checkpoints"):
        for filename in files:
            filepath = subdir + os.sep + filename
            W[filename[:-4]] = np.load(filepath)  # cut off '.npy'
    return W


def interpolate_verrs(arr):
    """ Interpolate missing validation accuracy and loss values.

    These exist, because validation may not happen in all epochs.
    """

    # TOOD: comments

    retarr = arr
    x0 = 0
    y0 = arr[x0]
    for idx, val in enumerate(arr):
        retarr[idx] = val

        if val != -1:
            x0 = idx
            y0 = val
            continue

        x1 = np.argmax(arr[idx:] > -1) + idx
        y1 = arr[x1]
        w = (idx - x0) / (x1 - x0)
        retarr[idx] = y0 * (1 - w) + y1 * w

    return retarr


def temporal_filter(c, a, depth=0):
    """ Low-pass filter of array `a' with smoothing factor `c'.

    The `depth' argument is used to limit the recusion depth.
    Consequentially, only the last 32 steps are considered.
    """

    if depth == 32:  # Maximal depth reached.
        return a[-1]
    if a.shape[0] == 1:  # Can't go back further -- no previous values
        return a[0]
    return c * temporal_filter(c, a=a[:-1], depth=depth+1) + a[-1:]


def normalize(arr):
    return np.interp(arr, (arr.min(), arr.max()), (-1, 1))


def eprop_Z(cfg, t, TZ, V, U, is_ALIF):
    return np.where(np.logical_and(t - TZ >= cfg["dt_refr"],
                                   V >= np.where(is_ALIF,
                                                 cfg["thr"] + cfg["beta"] * U,
                                                 cfg["thr"])),
                    1,
                    0)


def eprop_V(cfg, V, I, Z, t, TZ):
    if not cfg["traub_trick"]:
        return cfg["alpha"] * V + I - Z * cfg["thr"]
    else:
        # print(TZ.shape, (TZ == cfg["dt_refr"]).shape)
        return (cfg["alpha"] * V
                + I
                - cfg["alpha"] * Z * V
                - cfg["alpha"] * V * ((t - TZ) <= cfg["dt_refr"]))


def eprop_U(cfg, U, Z, is_ALIF):
    return np.where(is_ALIF,
                    cfg["rho"] * U + Z,
                    U)


def eprop_EVV(cfg, EVV, Z_in, t, TZ, TZ_in, Z):
    """
    ALIF & LIF: Zbar
    checked
    """
    if not cfg["traub_trick"]:
        return cfg["alpha"] * EVV + Z_in
    else:
        # Repeating to match EVV. Repeating axis differ because Z_in concerns presynaptics.
        Zrep = np.repeat(a=Z[:, np.newaxis], repeats=EVV.shape[1], axis=1)
        TZrep = np.repeat(a=TZ[:, np.newaxis], repeats=EVV.shape[1], axis=1)
        Z_inrep = np.repeat(a=Z_in[np.newaxis, :], repeats=EVV.shape[0], axis=0)
        TZ_inrep = np.repeat(a=TZ_in[np.newaxis, :], repeats=EVV.shape[0], axis=0)
        return (cfg["alpha"] * EVV * (1
                                - Zrep
                                - np.less_equal((t - TZ_inrep), cfg["dt_refr"])
                                - np.equal((t - TZrep), cfg["dt_refr"]))
                + Z_inrep)


def eprop_EVU(cfg, is_ALIF, H, Z_inbar, EVU):
    """
    ALIF: H_j * Zinbar_i + (rho - H_j * beta) * EVU_ji
    LIF: N/A
    checked
    """
    is_ALIF = np.repeat(a=is_ALIF[:, np.newaxis],
                        repeats=is_ALIF.size*2,
                        axis=1)

    Hp = cfg["rho"] - H * cfg["beta"]

    return np.where(is_ALIF,
                    np.outer(H, Z_inbar) + np.einsum("j, ji -> ji", Hp, EVU),
                    EVU)


def eprop_H(cfg, V, U, t, TZ, is_ALIF):  # TODO: 1/thr here? Traub and Bellec differ
    if cfg["traub_trick"]:
        return np.where(t - TZ < cfg["dt_refr"],
            -cfg["gamma"],
            cfg["gamma"] * np.clip(
            a=1 - (abs(V - np.where(is_ALIF,
                                    cfg["thr"] + cfg["beta"] * U,
                                    cfg["thr"]))
                       / cfg["thr"]),
            a_min=0,
            a_max=None))
    else:
        return 1 / cfg["thr"] * cfg["gamma"] * np.clip(
            a=1 - (abs(V - np.where(is_ALIF,
                                    cfg["thr"] + cfg["beta"] * U,
                                    cfg["thr"]))
                       / cfg["thr"]),
            a_min=0,
            a_max=None)


def eprop_ET(cfg, is_ALIF, H, EVV, EVU):
    """
    ET_ji = H_j * (EVV_ji - X)
    X = - beta * EVU_ji if ALIF else 0

    checked for LIF and ALIF!
    """
    is_ALIF = np.repeat(a=is_ALIF[:, np.newaxis],
                        repeats=is_ALIF.size*2,
                        axis=1)

    return np.where(is_ALIF,
                    np.einsum("j, ji->ji", H, EVV - cfg["beta"] * EVU),
                    np.einsum("j, ji->ji", H, EVV))


def eprop_lpfK(lpf, x, factor):
    return (factor * lpf) + x


def eprop_Y(cfg, Y, W_out, Z_last, b_out):
    return (cfg["kappa"] * Y
            + np.sum(W_out * Z_last, axis=1)
            + b_out)


def eprop_P(Y):
    ex = np.exp(Y - np.max(Y))
    return ex / np.sum(ex)


def eprop_CE(cfg, T, P, W_rec, W_out, B):
    # TODO: on which weights?
    W = np.concatenate((W_rec.flatten(),
                        W_out.flatten(),
                        B.flatten()))
    L2norm_W = np.linalg.norm(W) ** 2 * cfg["L2_reg"]

    return (-np.sum(T * np.log(1e-8 + P))
            + L2norm_W)


def eprop_gradient(wtype, L, ETbar, P, T, Zbar_last):
    if wtype == 'W':
        return np.einsum("rj,rji->rji", L, ETbar)
    elif wtype == 'W_out':
        return np.outer((P - T), Zbar_last)
    elif wtype == 'b_out':
        return P - T


def eprop_DW(cfg, wtype, s, adamvars, gradient, Zs, ET):  # TODO: No ETbar here?
    FR_reg = 0
    if wtype == 'W' and Zs.shape[0]:  # Add firing rate reg term
        FR_reg = (
            cfg["eta"]
            * cfg["FR_reg"]
            * np.mean(np.einsum("rj,rji->rji",
                                cfg["FR_target"] - np.mean(Zs, axis=0),
                                ET),
                      axis=0))

    if cfg["optimizer"] == 'SGD':
        return -cfg["eta"] * gradient + FR_reg

    elif cfg["optimizer"] == 'Adam':
        m = (adamvars["beta1"] * adamvars[f"m{wtype}"][s]
             + (1 - adamvars["beta1"]) * gradient)
        v = (adamvars["beta2"] * adamvars[f"v{wtype}"][s]
             + (1 - adamvars["beta2"]) * gradient ** 2)
        f1 = m / ( 1 - adamvars["beta1"])
        f2 = np.sqrt(v / (1 - adamvars["beta2"])) + adamvars["eps"]

        return cfg["eta"] * (f1 / f2) + FR_reg


def process_layer(cfg, M, t, s, r, W_rec):
    # Spike if V >= threshold
    M['Z'][s, t, r] = eprop_Z(cfg=cfg,
                              t=t,
                              TZ=M['TZ'][s, r],
                              V=M['V'][s, t, r],
                              U=M['U'][s, t, r],
                              is_ALIF=M['is_ALIF'][s, r])

    M['TZ'][s, r, M['Z'][s, t, r]==1] = t  # Log spike time

    # Pad any input with zeros to make it length N_R
    Z_prev = M['Z'][s, t, r-1] if r > 0 else \
        np.pad(M[f'X{s}'][t],
               (0, cfg["N_R"] - len(M[f'X{s}'][t])))

    TZ_prev = M['TZ'][s, r-1] if r > 0 else \
        np.ones(shape=(M['TZ'][s, r].shape)) * t

    M['H'][s, t, r] = eprop_H(cfg=cfg,
                              V=M['V'][s, t, r],
                              U=M['U'][s, t, r],
                              is_ALIF=M['is_ALIF'][s, r],
                              t=t,
                              TZ=M['TZ'][s, r])

    M['ET'][s, t, r] = eprop_ET(cfg=cfg,
                                is_ALIF=M['is_ALIF'][s, r],
                                H=M['H'][s, t, r],
                                EVV=M['EVV'][s, t, r],
                                EVU=M['EVU'][s, t, r])

    # Update weights for next epoch
    if not cfg["update_input_weights"]:
        for var in ["EVV", "EVU", "ET"]:
            M[var][s, t, 0, :, :M[f"X{s}"].shape[-1]] = 0

    # Update weights for next epoch
    if not cfg["update_dead_weights"]:
        for var in ["EVV", "EVU", "ET"]:
            M[var][s, t, r, W_rec == 0] = 0

    M["Z_in"][s, t, r] = np.concatenate((Z_prev, M['Z'][s, t, r]))
    M["TZ_in"][s, r] = np.concatenate((TZ_prev, M['TZ'][s, r]))

    M['Z_inbar'][s, t] = eprop_lpfK(lpf=M['Z_inbar'][s, t-1] if t > 0 else 0,
                                    x=M['Z_in'][s, t],
                                    factor=cfg["alpha"])

    M['I'][s, t, r] = np.dot(W_rec, M["Z_in"][s, t, r])

    M['ETbar'][s, t, r] = eprop_lpfK(lpf=M['ETbar'][s, t-1, r] if t > 0 else 0,
                                     x=M['ET'][s, t, r],
                                     factor=cfg["kappa"])

    M['Zbar'][s, t, r] = eprop_lpfK(lpf=M['Zbar'][s, t-1, r] if t > 0 else 0,
                                     x=M['Z'][s, t, r],
                                     factor=cfg["kappa"])

    if t != M[f"X{s}"].shape[0] - 1:
        M['EVV'][s, t+1, r] = eprop_EVV(cfg=cfg,
                                        EVV=M['EVV'][s, t, r],
                                        Z_in=M["Z_in"][s, t, r],
                                        Z=M['Z'][s, t, r],
                                        TZ=M['TZ'][s, r],
                                        TZ_in=M['TZ_in'][s, r],
                                        t=t)

        # TODO: Can do without M[ET] or M[H] or M[TZ] or M[DW].
        M['EVU'][s, t+1, r] = eprop_EVU(cfg=cfg,
                                        Z_inbar=M['Z_inbar'][s, t, r],
                                        EVU=M['EVU'][s, t, r],
                                        H=M['H'][s, t, r],
                                        is_ALIF=M['is_ALIF'][s, r])

        M['V'][s, t+1, r] = eprop_V(cfg=cfg,
                                    V=M['V'][s, t, r],
                                    I=M['I'][s, t, r],
                                    Z=M['Z'][s, t, r],
                                    TZ=M['TZ'][s, r],
                                    t=t)

        M['U'][s, t+1, r] = eprop_U(cfg=cfg,
                                    U=M['U'][s, t, r],
                                    Z=M['Z'][s, t, r],
                                    is_ALIF=M['is_ALIF'][s, r])


    return M


def init_adam(cfg, tar_size):
    return {
        'beta1': cfg['adam_beta1'],
        'beta2': cfg['adam_beta2'],
        'eps': cfg['adam_eps'],
        'mW': np.zeros(shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'vW': np.zeros(shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'mW_out': np.zeros(shape=(cfg["n_directions"], tar_size, cfg["N_R"],)),
        'vW_out': np.zeros(shape=(cfg["n_directions"], tar_size, cfg["N_R"],)),
        'mb_out': np.zeros(shape=(cfg["n_directions"], tar_size,)),
        'vb_out': np.zeros(shape=(cfg["n_directions"], tar_size,)),
    }


def prepare_log(cfg, log_id):
    log_subdirs = [
        'states',
        'checkpoints'
    ]

    for subdir in log_subdirs:
        os.makedirs(f"../log/{log_id}/{subdir}")

    cfg0 = dict(cfg)

    for k, v in cfg.items():
        if 'numpy' in str(type(v)):
            cfg0[k] = v.item()

    with open('config.json', 'w+') as fp:
        json.dump(cfg0, fp)

    # folder = f"../vis/states/"
    # for filename in os.listdir(folder):
    #     file_path = os.path.join(folder, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_log_id():
    log_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    id_append = ""
    count = 1
    while os.path.isdir(f"../log/{log_id}{id_append}"):
        id_append = f"_{count}"
        count += 1
    return log_id + id_append
