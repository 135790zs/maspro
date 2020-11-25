import os
import numpy as np
from config import cfg


def initialize_model(length, tar_size):
    rng = np.random.default_rng()
    M = {}
    neuron_shape = (cfg["n_directions"],
                    length,
                    cfg["N_Rec"],
                    cfg["N_R"],)
    weight_shape = (cfg["n_directions"],
                    length,
                    cfg["N_Rec"],
                    cfg["N_R"],
                    cfg["N_R"] * 2,)

    for neuronvar in ["V", "Z", "ZbarK", "I", "H", "L"]:
        M[neuronvar] = np.zeros(shape=neuron_shape)

    for weightvar in ["EVV", "EVU", "ET", "DW", "ETbar", 'gW']:
        M[weightvar] = np.zeros(shape=weight_shape)

    M["V"] = np.zeros(shape=neuron_shape)  # * cfg["thr"]
    M["U"] = np.ones(shape=neuron_shape) * cfg["thr"]
    M["TZ"] = np.ones(shape=(cfg["n_directions"],
                             cfg["N_Rec"],
                             cfg["N_R"])) * -cfg["dt_refr"]

    M["Z_in"] = np.zeros(shape=(cfg["n_directions"],
                                length,
                                cfg["N_Rec"],
                                cfg["N_R"] * 2,))
    M["Z_inbar"] = np.zeros(shape=(cfg["n_directions"],
                                   length,
                                   cfg["N_Rec"],
                                   cfg["N_R"] * 2,))

    M["DW_out"] = np.zeros(shape=(cfg["n_directions"],
                                  length,
                                  tar_size,
                                  cfg["N_R"],))
    M["gW_out"] = np.zeros(shape=(cfg["n_directions"],
                                  length,
                                  tar_size,
                                  cfg["N_R"],))
    M["DB"] = np.zeros(shape=(cfg["n_directions"],
                              length,
                              cfg["N_Rec"],
                              cfg["N_R"],
                              tar_size))
    M["Db_out"] = np.zeros(shape=(cfg["n_directions"],
                                  length,
                                  tar_size,))
    M["gb_out"] = np.zeros(shape=(cfg["n_directions"],
                                  length,
                                  tar_size,))

    M["Y"] = np.zeros(shape=(cfg["n_directions"], length, tar_size,))
    M["T"] = np.zeros(shape=(length, tar_size,))
    M["P"] = np.zeros(shape=(length, tar_size,))
    M["Pmax"] = np.zeros(shape=(length, tar_size,))
    M["CE"] = np.zeros(shape=(length,))

    M["is_ALIF"] = np.zeros(
        shape=(cfg["n_directions"] * cfg["N_Rec"] * cfg["N_R"]))
    M["is_ALIF"][:int(M["is_ALIF"].size * cfg["fraction_ALIF"])] = 1
    np.random.shuffle(M["is_ALIF"])
    M["is_ALIF"] = M["is_ALIF"].reshape((cfg["n_directions"],
                                         cfg["N_Rec"],
                                         cfg["N_R"]))

    return M


def initialize_weights(tar_size):
    rng = np.random.default_rng()
    W = {}

    W["W"] = rng.random(
        size=(cfg["Epochs"], cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,))
    W["W"][0, :, 0] /= cfg["N_R"]
    # W["W"][0, :, 1] /= (cfg["N_R"]) / 2
    # W["W"][0, 2] /= 32
    W["W_out"] = rng.random(size=(cfg["Epochs"], cfg["n_directions"], tar_size, cfg["N_R"],))
    W["b_out"] = np.zeros(shape=(cfg["Epochs"], cfg["n_directions"], tar_size,))

    if cfg["eprop_type"] == "random":  # Variance of 1
        W["B"] = rng.normal(
            size=(cfg["Epochs"], cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], tar_size,),
            scale=1)
    elif cfg["eprop_type"] == "adaptive":  # Variance of 1/N
        W["B"] = rng.normal(
            size=(cfg["Epochs"], cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], tar_size,),
            scale=np.sqrt(1/cfg["N_R"]))
    else:
        W["B"] = rng.random(
            size=(cfg["Epochs"], cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], tar_size,))

    for r in range(cfg["N_Rec"]):
        for s in range(cfg["n_directions"]):
        # Zero diag recurrent W: no self-conn
            np.fill_diagonal(W['W'][0, s, r, :, cfg["N_R"]:], 0)

    # W['W'][0, 0, 0, 0] = 0.8  # Input 1 to rec 1: frozen
    # W['W'][0, 0, 1, 0] = 0  # Input 1 to rec 2
    # W['W'][0, 0, 0, 1] = 0  # Input 2 to rec 1
    # W['W'][0, 0, 1, 1] = 0.8  # Input 2 to rec 2: frozen
    # W['W'][0, 0, 1, 2] = 0.2  # Rec 1 to rec 2 (B)
    # W['W'][0, 0, 0, 3] = 0.2  # Rec 2 to rec 1 (T)


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


def interpolate_verrs(arr):
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


def eprop_Zinbar(Z_inbar, Z_in):
    return ((cfg["alpha"] * M['Z_inbar'][t-1] if t > 0 else 0)
                               + M['Z_in'][t])


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
    """
    ALIF & LIF: Zbar
    checked
    """
    return cfg["alpha"] * EVV + Z_in


def eprop_EVU(is_ALIF, H, Z_inbar, EVU):
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


def eprop_H(V, U, is_ALIF):
    return 1 / cfg["thr"] * \
        cfg["gamma"] * np.clip(a=1 - (abs(V
                                          - np.where(is_ALIF, U, cfg["thr"])
                                          )
                                      / cfg["thr"]),
                               a_min=0,
                               a_max=None)


def eprop_ET(is_ALIF, H, EVV, EVU):
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


def eprop_Y(Y, W_out, Z_last, b_out):
    return (cfg["kappa"] * Y
            + np.sum(W_out * Z_last, axis=1)
            + b_out)


def eprop_P(Y):
    ex = np.exp(Y - np.max(Y))
    return ex / np.sum(ex)


def eprop_CE(T, P, W_rec, W_out, B):
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


def eprop_DW(wtype, s, adamvars, gradient, Zs, ET):
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


def process_layer(M, t, s, r, W_rec):
    # Spike if V >= threshold
    M['Z'][s, t, r] = eprop_Z(t=t,
                              TZ=M['TZ'][s, r],
                              V=M['V'][s, t, r],
                              U=M['U'][s, t, r])

    M['TZ'][s, r, M['Z'][s, t, r]==1] = t  # Log spike time

    # Pad any input with zeros to make it length N_R
    Z_prev = M['Z'][s, t, r-1] if r > 0 else \
        np.pad(M[f'X{s}'][t],
               (0, cfg["N_R"] - len(M[f'X{s}'][t])))

    M['H'][s, t, r] = eprop_H(V=M['V'][s, t, r],
                              U=M['U'][s, t, r],
                              is_ALIF=M['is_ALIF'][s, r])

    M['ET'][s, t, r] = eprop_ET(is_ALIF=M['is_ALIF'][s, r],
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

    M['Z_inbar'][s, t] = eprop_lpfK(lpf=M['Z_inbar'][s, t-1] if t > 0 else 0,
                                    x=M['Z_in'][s, t],
                                    factor=cfg["alpha"])

    M['I'][s, t, r] = np.dot(W_rec, M["Z_in"][s, t, r])

    M['ETbar'][s, t, r] = eprop_lpfK(lpf=M['ETbar'][s, t-1, r] if t > 0 else 0,
                                     x=M['ET'][s, t, r],
                                     factor=cfg["kappa"])

    M['ZbarK'][s, t, r] = eprop_lpfK(lpf=M['ZbarK'][s, t-1, r] if t > 0 else 0,
                                     x=M['Z'][s, t, r],
                                     factor=cfg["kappa"])

    if t != M[f"X{s}"].shape[0] - 1:
        M['EVV'][s, t+1, r] = eprop_EVV(EVV=M['EVV'][s, t, r],
                                        Z_in=M["Z_in"][s, t, r])

        # TODO: Can do without M[ET] or M[H] or M[TZ] or M[DW].
        M['EVU'][s, t+1, r] = eprop_EVU(Z_inbar=M['Z_inbar'][s, t, r],
                                        EVU=M['EVU'][s, t, r],
                                        H=M['H'][s, t, r],
                                        is_ALIF=M['is_ALIF'][s, r])

        M['V'][s, t+1, r] = eprop_V(V=M['V'][s, t, r],
                                    I=M['I'][s, t, r],
                                    Z=M['Z'][s, t, r])

        M['U'][s, t+1, r] = eprop_U(U=M['U'][s, t, r],
                                    Z=M['Z'][s, t, r],
                                    is_ALIF=M['is_ALIF'][s, r])


    return M



def init_adam(tar_size):
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
