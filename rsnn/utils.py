import time
import os
import numpy as np
import datetime
import json
from scipy.interpolate import interp1d
# import cupy as cp


def initialize_model(cfg, length, tar_size):
    """ Initializes the variables used in predicting a single time series.

        These are the variables used in e-prop, and the reason they're
        stored for every time step (rather than overwriting per time step)
        is so the evolution of the state can be inspected after running.

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
        ET:      Loww-pass filter of ynapse eligibility trace
        gW:      Network weights gradient
        DW:      Network weights update
        dW_out:  Output weights update
        gW_out:  Output weights gradient
        db_out:  Bias update
        gb_out:  Bias gradient
        T:       Timestep target
        Y:       Timestep output
        P:       Timestep probability vector
        D:       Timestep probability minus target
        Pmax:    Timestep prediction
        CE:      Timestep cross-entropy loss
        L:       Timestep learning signal
        is_ALIF: Mask determining which neurons have adaptive thresholds
    """

    pruned_length = length if cfg["Track_synapse"] else 1

    rng = np.random.default_rng(seed=cfg["seed"])

    M = {}
    neuron_shape = (cfg["n_directions"],
                    length,
                    cfg["N_Rec"],
                    cfg["N_R"],)

    neuron_timeless_shape = (cfg["n_directions"],
                             cfg["N_Rec"],
                             cfg["N_R"],)

    weight_shape = (cfg["n_directions"],
                    pruned_length,
                    cfg["N_Rec"],
                    cfg["N_R"],
                    cfg["N_R"] * 2,)
    Z_in_shape = (cfg["n_directions"],
                  length,
                  cfg["N_Rec"],
                  cfg["N_R"] * 2,)

    W_out_shape = (cfg["n_directions"],
                   pruned_length,
                   tar_size,
                   cfg["N_R"],)

    b_out_shape = (cfg["n_directions"],
                   pruned_length,
                   tar_size,)

    T_shape = (length, tar_size,)

    for T_var in ["T", "P", "Pmax", "D"]:
        M[T_var] = np.zeros(shape=T_shape)

    for neuronvar in ["Z", "V", "Zbar", "Z_prev", "I", "L_std", "L_reg", "spikerate"]:
        M[neuronvar] = np.zeros(shape=neuron_shape)

    for weightvar in ["EVV", "EVU", "ET", "ETbar", 'gW']:
        M[weightvar] = np.zeros(shape=weight_shape)

    for z_in_var in ["Z_in", "Z_inbar"]:
        M[z_in_var] = np.zeros(shape=Z_in_shape)

    for W_out_var in ["gW_out"]:
        M[W_out_var] = np.zeros(shape=W_out_shape)

    for b_out_var in ["gb_out"]:
        M[b_out_var] = np.zeros(shape=b_out_shape)

    M["U"] = np.ones(shape=neuron_shape) * cfg["thr"]
    # M["V"] = rng.random(size=neuron_shape) * cfg["thr"]  # TODO: Revert
    M["H"] = rng.random(size=neuron_shape) * cfg["gamma"]

    M["TZ"] = np.ones(shape=neuron_timeless_shape) * -cfg["dt_refr"]

    M["TZ_in"] = np.zeros(shape=(cfg["n_directions"],
                                 cfg["N_Rec"],
                                 cfg["N_R"] * 2,))

    M["Y"] = np.zeros(shape=(cfg["n_directions"], length, tar_size,))
    M["CE"] = np.zeros(shape=(length,))

    M["is_ALIF"] = np.zeros(
        shape=(cfg["n_directions"] * cfg["N_Rec"] * cfg["N_R"]))

    M["is_ALIF"][:int(M["is_ALIF"].size * cfg["fraction_ALIF"])] = 1
    rng.shuffle(M["is_ALIF"])

    M["is_ALIF"] = M["is_ALIF"].reshape(neuron_timeless_shape)
    return M


def initialize_weights(cfg, inp_size, tar_size):
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
    rng = np.random.default_rng(seed=cfg["seed"])

    n_epochs = cfg["Epochs"] if cfg["Track_weights"] else 1
    if cfg["uniform_weights"]:
        W["W"] = rng.random(size=(n_epochs,
                                  cfg["n_directions"],
                                  cfg["N_Rec"],
                                  cfg["N_R"],
                                  cfg["N_R"] * 2)) * 2 - 1

        W["W_out"] = rng.random(size=(n_epochs,
                                      cfg["n_directions"],
                                      tar_size,
                                      cfg["N_R"])) * 2 - 1
    else:
        W["W"] = rng.normal(size=(n_epochs,
                                  cfg["n_directions"],
                                  cfg["N_Rec"],
                                  cfg["N_R"],
                                  cfg["N_R"] * 2,),
                            scale=0.5)

        W["W_out"] = rng.normal(size=(n_epochs,
                                      cfg["n_directions"],
                                      tar_size,
                                      cfg["N_R"],),
                                scale=0.25)

    if cfg["one_to_one_output"]:
        for s in range(cfg["n_directions"]):
            W["W_out"][0, s] = 0
            np.fill_diagonal(W["W_out"][0, s, :, tar_size:], 1)

    W["b_out"] = rng.random(size=(n_epochs,
                                 cfg["n_directions"],
                                 tar_size,)) * 2 - 1

    B_shape = (n_epochs,
               cfg["n_directions"],
               cfg["N_Rec"],
               cfg["N_R"],
               tar_size,)

    if cfg["eprop_type"] == "random":  # Gaussian, variance of 1
        W["B"] = rng.normal(size=B_shape, scale=1)

    elif cfg["eprop_type"] == "global":  # Gaussian, variance of 1
        W["B"] = np.ones(shape=B_shape) * (1/np.sqrt(cfg["N_R"]*cfg["N_Rec"]))

    elif cfg["eprop_type"] == "adaptive":  # Gaussian, variance of 1/N
        W["B"] = rng.normal(size=B_shape, scale=np.sqrt(1 / cfg["N_R"]))

    else:  # Symmetric: Uniform [0, 1]. Irrelevant, as it'll be overwritten
        W["B"] = rng.random(size=B_shape) * 2 - 1

    # Drop all self-looping weights. A neuron cannot be connected with itself.
    for r in range(cfg["N_Rec"]):
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W['W'][0, s, r, :, cfg["N_R"]:], 0)

    # Randomly dropout a fraction of the (remaining) recurrent weights.
    # inps = W['W'][0]
    W['W'][0] = np.where(rng.random(W['W'][0].shape) < cfg["dropout"],
                         0,
                         W['W'][0])
    # W['W'][0] = inps

    if cfg["one_to_one_input"]:
        W['W'][0, :, 0, :, :inp_size] = 0
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W['W'][0, s, 0, :, :inp_size], 1)

    W['W'][0, :, 0] *= cfg["weight_scaling"]

    return W


def initialize_DWs(cfg, inp_size, tar_size):
    DW = {}
    DW["W"] = np.zeros(shape=(cfg["n_directions"],
                              cfg["N_Rec"],
                              cfg["N_R"],
                              cfg["N_R"] * 2,))

    DW["W_out"] = np.zeros(shape=(cfg["n_directions"],
                                  tar_size,
                                  cfg["N_R"],))

    DW["b_out"] = np.zeros(shape=(cfg["n_directions"],
                                 tar_size,))
    return DW


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
    # if cfg["fraction_ALIF"] == 1:
    #     thr = cfg["thr"] + cfg["beta"] * U
    # elif cfg["fraction_ALIF"] == 0:
    #     thr = cfg["thr"]
    # else:
    #     thr = np.where(is_ALIF,
    #                    cfg["thr"] + cfg["beta"] * U,
    #                    cfg["thr"])

    return np.where(np.logical_and(t - TZ >= cfg["dt_refr"],
                                   V >= cfg["thr"] + cfg["beta"] * U),
                    1,
                    0)


def eprop_I(W_rec, Z_in):
    return np.dot(W_rec, Z_in)


def eprop_V(cfg, V, I, Z, U, t, TZ):
    if not cfg["traub_trick"]:
        # if cfg["fraction_ALIF"] == 1:
        #     thr = cfg["thr"] + cfg["beta"] * U
        # elif cfg["fraction_ALIF"] == 0:
        #     thr = cfg["thr"]
        # else:
        #     thr = np.where(is_ALIF,  # TODO: Think about this!
        #                    cfg["thr"] + cfg["beta"] * U,
        #                    cfg["thr"])
        return cfg["alpha"] * V + I - Z * (cfg["thr"] + cfg["beta"] * U)
    else:
        return (cfg["alpha"] * V
                + I
                - cfg["alpha"] * Z * V
                - cfg["alpha"] * V * ((t - TZ) <= cfg["dt_refr"]))


def eprop_U(cfg, U, Z, is_ALIF):

    return U + is_ALIF * ((cfg["rho"]) * U + Z)
    # if cfg["fraction_ALIF"] == 1:
    #     return cfg["rho"] * U + Z
    # elif cfg["fraction_ALIF"] == 0:
    #     return U
    # return np.where(is_ALIF,
    #                 cfg["rho"] * U + Z,
    #                 U)


def eprop_EVV(cfg, EVV, Z_in, t, TZ, TZ_in, Z):
    """
    ALIF & LIF: Zbar
    checked
    """
    if not cfg["traub_trick"]:
        return cfg["alpha"] * EVV + Z_in
    else:
        print("WARNING: UNCOMMENT TZ_IN")
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


# def eprop_EVU(cfg, is_ALIF, H, Z_inbar, EVU):
#     """
#     ALIF: H_j * Zinbar_i + (rho - H_j * beta) * EVU_ji
#     LIF: N/A
#     checked
#     """
#     # Hp = cfg["rho"] - H * cfg["beta"]

#     # if cfg["fraction_ALIF"] == 1:
#     #     return np.outer(H, Z_inbar) + np.einsum("j, ji -> ji", Hp, EVU)
#     # elif cfg["fraction_ALIF"] == 0:
#     #     return EVU

#     # is_ALIF = np.repeat(a=is_ALIF[:, np.newaxis],
#     #                     repeats=is_ALIF.size*2,
#     #                     axis=1)

#     # ret = np.where(is_ALIF,
#     #                np.outer(H, Z_inbar) + np.einsum("j, ji -> ji", Hp, EVU),
#     #                EVU)

#     return (np.outer(H, Z_inbar)
#             + np.einsum("j, ji -> ji",
#                         cfg["rho"] - H * cfg["beta"],
#                         EVU)
#             ) * np.repeat(a=is_ALIF[:, np.newaxis],
#                           repeats=is_ALIF.size*2,
#                           axis=1)


def eprop_H(cfg, V, U, t, TZ, is_ALIF):  # TODO: 1/thr here? Traub and Bellec differ
    # if cfg["fraction_ALIF"] == 1:
    #     thr = cfg["thr"] + cfg["beta"] * U
    # elif cfg["fraction_ALIF"] == 0:
    #     thr = cfg["thr"]
    # else:
    #     thr = np.where(is_ALIF,
    #                    cfg["thr"] + cfg["beta"] * U,
    #                    cfg["thr"])

    if cfg["traub_trick"]:
        return np.where(t - TZ < cfg["dt_refr"],
            -cfg["gamma"],
            cfg["gamma"] * np.clip(
            a=1 - (abs(V - (cfg["thr"] + cfg["beta"] * U)) / cfg["thr"]),
            a_min=0,
            a_max=None))
    else:
        return 1 / cfg["thr"] * cfg["gamma"] * np.clip(
            a=1 - (abs(V - (cfg["thr"] + cfg["beta"] * U)) / cfg["thr"]),
            a_min=0,
            a_max=None)


# def eprop_ET(cfg, is_ALIF, H, EVV, EVU):
#     """
#     ET_ji = H_j * (EVV_ji - X)
#     X = - beta * EVU_ji if ALIF else 0

#     checked for LIF and ALIF!
#     """

#     # if cfg["fraction_ALIF"] == 1:
#     #     return np.einsum("j, ji->ji", H, EVV - cfg["beta"] * EVU)
#     # elif cfg["fraction_ALIF"] == 0:
#     #     return np.einsum("j, ji->ji", H, EVV)

#     # is_ALIF = np.repeat(a=is_ALIF[:, np.newaxis],
#     #                     repeats=is_ALIF.size*2,
#     #                     axis=1)

#     # ret = np.where(is_ALIF,  # Timesink!
#     #                np.einsum("j, ji->ji", H, EVV - cfg["beta"] * EVU),
#     #                np.einsum("j, ji->ji", H, EVV))

#     return np.einsum("j, ji->ji", H, EVV - cfg["beta"] * EVU)


def eprop_Y(cfg, Y, W_out, Z_last, b_out):
    return (cfg["kappa"] * Y
            + np.sum(W_out * Z_last, axis=1)
            + b_out)


def eprop_P(cfg, Y):
    """ Softmax"""
    maxx = np.max(Y)
    m = Y - maxx
    ex = np.exp(cfg["softmax_factor"] * m)
    denom = np.sum(ex)
    ret = ex / denom
    return ret


# def eprop_CE(cfg, T, P, W_rec, W_out, B):
#     """ Has no effect on training, only serves as performance metric. """

#     return -np.sum(T * np.log(1e-30 + P))


# def eprop_gradient(wtype, L_std, L_reg, ETbar, D, Zbar_last):
#     """ Return the gradient of the weights. """
#     if wtype == "W":
#         return (np.einsum("rj,rji->rji", L_std, ETbar)  # Checked correct
#                 + np.repeat(L_reg[:, :, np.newaxis],
#                             repeats=L_std.shape[-1]*2,
#                             axis=2))  # Checked correct
#     elif wtype == "W_out":
#         return np.outer(D, Zbar_last)
#     elif wtype == "b_out":
#         return D


# def eprop_DW(cfg, wtype, s, adamvars, gradient, eta):
#     eta = cfg["eta_b_out"] if wtype == "b_out" and cfg["eta_b_out"] is not None else eta

#     if cfg["optimizer"] == 'SGD':
#         return -eta * gradient
#     elif cfg["optimizer"] == 'Adam':
#         m = (cfg["adam_beta1"] * adamvars[f"m{wtype}"][s]
#              + (1 - cfg["adam_beta1"]) * gradient)
#         v = (cfg["adam_beta2"] * adamvars[f"v{wtype}"][s]
#              + (1 - cfg["adam_beta2"]) * (gradient ** 2))
#         f1 = m / ( 1 - cfg["adam_beta1"])
#         f2 = np.sqrt(v / (1 - cfg["adam_beta2"])) + cfg["adam_eps"]
#         ret = -eta * (f1 / f2)
#         return ret


def process_layer(cfg, M, t, s, r, W_rec):
    """ Process a single layer of the model at a single time step."""

    # If not tracking state, the time dimensions of synaptic variables have
    # length 1 and we overwrite those previous time steps at index 0.
    if cfg["Track_synapse"]:
        prev_t = t-1
        curr_t = t
        next_t = t+1
    else:
        prev_t = 0
        curr_t = 0
        next_t = 0

    # Pad any input with zeros to make it length N_R. Used to be able to dot
    # product with weights.
    M["Z_prev"][s, t, r] = M['Z'][s, t, r-1] if r > 0 else \
        np.pad(M[f'X{s}'][t],
               (0, cfg["N_R"] - M[f'X{s}'].shape[1]))

    M["Z_in"][s, t, r] = np.concatenate((M["Z_prev"][s, t, r], M['Z'][s, t-1, r]))

    # Revert eprop functions EVV and V if using traub trick!
    assert(not cfg["traub_trick"])

    # EVV
    M['EVV'][s, curr_t, r] = (cfg["alpha"] * M['EVV'][s, prev_t, r]
                              + M["Z_in"][s, t-1, r])

    # EVU
    M['EVU'][s, curr_t, r] = (np.outer(M['H'][s, prev_t, r],
                              M['Z_inbar'][s, t-1, r])
                              + np.einsum("j, ji -> ji",
                                          (cfg["rho"]
                                           - (M['H'][s, prev_t, r]
                                              * cfg["beta"])),
                                          M['EVU'][s, prev_t, r])
                              ) * np.repeat(
                                  a=M['is_ALIF'][s, r][:, np.newaxis],
                                  repeats=M['is_ALIF'][s, r].size*2,
                                  axis=1)


    # I
    M['I'][s, t, r] = np.dot(W_rec, M["Z_in"][s, t, r])

    # V
    M['V'][s, t, r] =  cfg["alpha"] * M['V'][s, t-1, r] + M['I'][s, t, r] - M['Z'][s, t-1, r] * (cfg["thr"] + cfg["beta"] * M['U'][s, t-1, r])

    # U
    M['U'][s, t, r] = (M['U'][s, t-1, r]
                         + (M['is_ALIF'][s, r]
                            * ((cfg["rho"] - 1) * M['U'][s, t-1, r]
                               + M['Z'][s, t-1, r])))

    # Update the spikes Z of the neurons.
    M['Z'][s, t, r] = np.where(
        np.logical_and(t - M['TZ'][s, r] >= cfg["dt_refr"],
                       M['V'][s, t, r] >= (cfg["thr"]
                                           + cfg["beta"] * M['U'][s, t, r])),
        1,
        0)

    # TZ is time of latest spike
    M['TZ'][s, r, M['Z'][s, t, r]==1] = t

    # Pseudoderivative H
    M['H'][s, t, r] = eprop_H(cfg=cfg,
                              V=M['V'][s, t, r],
                              U=M['U'][s, t, r],
                              is_ALIF=M['is_ALIF'][s, r],
                              t=t,
                              TZ=M['TZ'][s, r])

    # ET

    M['ET'][s, curr_t, r] = np.einsum("j, ji->ji",
                                      M['H'][s, t, r],
                                      (M['EVV'][s, curr_t, r]
                                       - cfg["beta"] * M['EVU'][s, curr_t, r]))


    # Input layer always "spikes" (with nonbinary spike values Z=X).
    # TZ_prev = M['TZ'][s, r-1] if r > 0 else \
    #     np.ones(shape=(M['TZ'][s, r].shape)) * t

    # M["TZ_in"][s, r] = np.concatenate((TZ_prev, M['TZ'][s, r]-1))

    M['Z_inbar'][s, t] = (cfg["alpha"] * (M['Z_inbar'][s, t-1] if t > 0 else 0)) + M['Z_in'][s, t]
    M['ETbar'][s, curr_t, r] = cfg["kappa"] * (M['ETbar'][s, prev_t, r] if t > 0 else 0) + M['ET'][s, curr_t, r]
    M['Zbar'][s, t, r] = cfg["kappa"] * (M['Zbar'][s, t-1, r] if t > 0 else 0) + M['Z'][s, t, r]

    return M


def init_adam(cfg, tar_size):
    return {
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

    with open(f'../log/{log_id}/config.json', 'w+') as fp:
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


def get_elapsed_time(cfg, start_time, e, b, batch_size):
    plustime = time.strftime('%H:%M:%S',
                             time.gmtime(time.time()-start_time))
    remtime = ''
    if e:
        timeperbatch = ((time.time() - start_time)
                        / (e * batch_size + b))

        remainingbatches = (cfg["Epochs"] - e) * batch_size + (batch_size - b)

        if cfg["val_every_E"]:
            remainingbatches += remainingbatches * (1/cfg["val_every_E"])

        remtime = '\t-' + time.strftime('%H:%M:%S', time.gmtime(
            timeperbatch * remainingbatches))

    return plustime, remtime


def eprop_FR_reg(cfg, rates, ETbar, t):
    ret = eta * cfg["FR_reg"] * np.sum()
    return ret


def interpolate_inputs(inp, tar, stretch):
    inp = inp.T
    tar = tar.T

    itp_inp = interp1d(np.arange(inp.shape[-1]),
                   inp,
                   kind='linear')

    itp_tar = interp1d(np.arange(tar.shape[-1]),
                   tar,
                   kind='nearest')

    inp = itp_inp(np.linspace(0, inp.shape[-1]-1, int(inp.shape[-1]*stretch)))
    tar = itp_tar(np.linspace(0, tar.shape[-1]-1, int(tar.shape[-1]*stretch)))

    return inp.T, tar.T

    # this_inps = np.repeat(this_inps, cfg["Repeats"], axis=0)
    # this_tars = np.repeat(this_tars, cfg["Repeats"], axis=0)
