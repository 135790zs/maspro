import time
import os
import numpy as np
from numba import jit, cuda, vectorize
# import pytorch as torch
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

    for neuronvar in ["Z", "V", "a", "H", "Zbar", "Z_prev", "I", "I_in", "I_rec", "L_std", "L_reg", "spikerate"]:
        M[neuronvar] = np.zeros(shape=neuron_shape)

    for weightvar in ["EVV", "EVU", "ET", "ETbar", 'gW']:
        M[weightvar] = np.zeros(shape=weight_shape)

    for z_in_var in ["Z_in", "Z_inbar"]:
        M[z_in_var] = np.zeros(shape=Z_in_shape)

    for W_out_var in ["gW_out"]:
        M[W_out_var] = np.zeros(shape=W_out_shape)

    for b_out_var in ["gb_out"]:
        M[b_out_var] = np.zeros(shape=b_out_shape)

    M["A"] = np.ones(shape=neuron_shape) * cfg["thr"]

    M["TZ"] = np.ones(shape=neuron_timeless_shape) * -cfg["dt_refr"]

    M["TZ_in"] = np.zeros(shape=(cfg["n_directions"],
                                 cfg["N_Rec"],
                                 cfg["N_R"] * 2,))

    M["Y"] = np.zeros(shape=(cfg["n_directions"], length, tar_size,))
    M["CE"] = np.zeros(shape=(length,))
    M["Correct"] = np.zeros(shape=(length,))

    return M

def initialize_betas(cfg):
    rng = np.random.default_rng(seed=cfg["seed"])
    betas = np.zeros(
        shape=(cfg["n_directions"] * cfg["N_Rec"] * cfg["N_R"]))

    betas[:int(betas.size * cfg["fraction_ALIF"])] = cfg["beta"]
    rng.shuffle(betas)

    betas = betas.reshape((cfg["n_directions"],
                             cfg["N_Rec"],
                             cfg["N_R"],))
    return betas


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
    if cfg["weight_initialization"] == 'uniform':
        W["W"] = rng.random(size=(n_epochs,
                                  cfg["n_directions"],
                                  cfg["N_Rec"],
                                  cfg["N_R"],
                                  cfg["N_R"] * 2)) * 2 - 1

        W["W_out"] = rng.random(size=(n_epochs,
                                      cfg["n_directions"],
                                      tar_size,
                                      cfg["N_R"])) * 2 - 1

    elif cfg["weight_initialization"] == 'normal':
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

    elif cfg["weight_initialization"] == 'bellec18':
        """ See Supplementary information for: Long short-term
        memory and learning-to-learn in networks of spiking
        neurons """
        W["W"] = rng.normal(size=(n_epochs,
                                  cfg["n_directions"],
                                  cfg["N_Rec"],
                                  cfg["N_R"],
                                  cfg["N_R"] * 2,))
        # W['W'][0, :, :, :, cfg["N_R"]:] /= np.sqrt(cfg["N_R"])
        # W['W'][0, :, :, :, :cfg["N_R"]] /= np.sqrt(cfg["N_R"])
        W['W'][0] /= np.sqrt(cfg["N_R"]+inp_size)
        W["W_out"] = rng.normal(size=(n_epochs,
                                      cfg["n_directions"],
                                      tar_size,
                                      cfg["N_R"])) / np.sqrt(cfg["N_R"])

    W["b_out"] = np.zeros(shape=(n_epochs,
                                 cfg["n_directions"],
                                 tar_size,))

    if cfg["one_to_one_output"]:
        assert tar_size <= cfg["N_R"]
        W["W_out"][0] = 0
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W["W_out"][0, s, :, :], 1)


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
        for r in range(cfg["N_Rec"]):
            for s in range(cfg["n_directions"]):
                W["B"][0, s, r] = W["W_out"][0, s].T

    # Drop all self-looping weights. A neuron cannot be connected with itself.
    for r in range(cfg["N_Rec"]):
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W['W'][0, s, r, :, cfg["N_R"]:], 0)

    # Randomly dropout a fraction of the (remaining) recurrent weights.
    # inps = W['W'][0]
    W['W'][0] = np.where(rng.random(W['W'][0].shape) < cfg["dropout"],
                         0,
                         W['W'][0])

    if cfg["one_to_one_input"]:
        W['W'][0, :, 0, :, :inp_size] = 0
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W['W'][0, s, 0, :, :inp_size], 1)

    W['W'][0, :, 0] *= cfg["weight_scaling"]

    if cfg["load_checkpoints"]:
        checkpoint = load_weights(log_id=cfg["load_checkpoints"], parent_dir='vault')
        for wtype, arr in checkpoint.items():
            W[wtype][0] = arr

    if not cfg['recurrent']:
        W['W'][0, :, :, :, cfg["N_R"]:] = 0

    return W


def initialize_DWs(cfg, tar_size):
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


def gW_tracker(cfg, tar_size):
    GW = {}
    GW["W"] = np.zeros(shape=(cfg["Epochs"],
                              cfg["n_directions"],
                              cfg["N_Rec"],
                              cfg["N_R"],
                              cfg["N_R"] * 2,))

    GW["W_out"] = np.zeros(shape=(cfg["Epochs"],
                                  cfg["n_directions"],
                                  tar_size,
                                  cfg["N_R"],))

    GW["b_out"] = np.zeros(shape=(cfg["Epochs"],
                                  cfg["n_directions"],
                                 tar_size,))
    return GW

def initialize_tracking(cfg):
    R = {'err': {}, '%wrong': {}, 'Hz': {}}
    for tv_type in ['train', 'val']:
        R['err'][tv_type] = np.ones(shape=(cfg["Epochs"])) * -1
        R[f'%wrong'][tv_type] = np.ones(shape=(cfg["Epochs"])) * -1
    R['Hz'] = np.ones(shape=(cfg["Epochs"],
                             cfg["n_directions"],
                             cfg["N_Rec"],
                             cfg["N_R"])) * -1
    R['eta'] = np.ones(shape=(cfg["Epochs"])) * -1
    R['latest_val_err'] = None
    R['optimal_val_err'] = None


    return R


def save_weights(W, epoch, log_id):
    """ Saves well-performing weights to files, to be loaded later.

    Every individual run gets its own weight checkpoints.
    """

    for k, v in W.items():
        # Every type of weight (W, W_out, b_out, B) gets its own file.
        np.save(f"../log/{log_id}/checkpoints/{k}", v[epoch])


def load_weights(log_id, parent_dir='log'):
    """ Load weights from file to be used in network again.

    Mainly used for testing purposes, because the weights corresponding
    to the lowest validation loss are stored.
    """

    W = {}
    for subdir, _, files in os.walk(f"../{parent_dir}/{log_id}/checkpoints"):
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


# def eprop_EVV(cfg, EVV, Z_in, t, TZ, TZ_in, Z):
#     """
#     ALIF & LIF: Zbar
#     checked
#     """
#     if not cfg["traub_trick"]:
#         return cfg["alpha"] * EVV + Z_in
#     else:
#         print("WARNING: UNCOMMENT TZ_IN")
#         # Repeating to match EVV. Repeating axis differ because Z_in concerns presynaptics.
#         Zrep = np.repeat(a=Z[:, np.newaxis], repeats=EVV.shape[1], axis=1)
#         TZrep = np.repeat(a=TZ[:, np.newaxis], repeats=EVV.shape[1], axis=1)
#         Z_inrep = np.repeat(a=Z_in[np.newaxis, :], repeats=EVV.shape[0], axis=0)
#         TZ_inrep = np.repeat(a=TZ_in[np.newaxis, :], repeats=EVV.shape[0], axis=0)
#         return (cfg["alpha"] * EVV * (1
#                                 - Zrep
#                                 - np.less_equal((t - TZ_inrep), cfg["dt_refr"])
#                                 - np.equal((t - TZrep), cfg["dt_refr"]))
#                 + Z_inrep)

def eprop_P(cfg, Y):
    """ Softmax"""
    maxx = np.max(Y)
    m = Y - maxx
    ex = np.exp(cfg["softmax_factor"] * m)
    denom = np.sum(ex)
    ret = ex / denom
    return ret


# # @cuda.jit
# # @vectorize(['float64(float64, float64)'], target='cuda')
def einsum(a,b):
    return (a*b.T).T
    # ac = cp.asarray(a)
    # bc = cp.asarray(b)
    # return cp.asnumpy((a*b.T).T)
    # return np.einsum("j, ji -> i", a, b, optimize='optimal')
    # return



# @cuda.jit
# def ET(H, V, B, U, E):
#     # Define an array in the shared memory
#     # The size and type of the arrays must be known at compile time
#     sH = cuda.shared.array(shape=(NR,), dtype=float32)
#     sV = cuda.shared.array(shape=(NR, NR*2), dtype=float32)
#     sB = cuda.shared.array(shape=(NR,), dtype=float32)
#     sU = cuda.shared.array(shape=(NR, NR*2), dtype=float32)

#     x, y = cuda.grid(2)

#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#     bpg = cuda.gridDim.x    # blocks per grid

#     if x >= E.shape[0] and y >= E.shape[1]:
#         # Quit if (x, y) is outside of valid E boundary
#         return

#     # Each thread computes one element in the result matrix.
#     # The dot product is chunked into dot products of NR-long vectors.
#     tmp = 0.
#     for i in range(bpg):
#         # Preload data into shared memory
#         sH[tx] = H[x]
#         sB[tx] = B[x]
#         sV[tx, ty] = V[tx + i * NR, y]
#         sU[tx, ty] = U[tx + i * NR, y]

#         # Wait until all threads finish preloading
#         cuda.syncthreads()

#         # Computes partial product on the shared memory
#         for j in range(NR):
#             tmp += sA[tx, j] * sB[j, ty]

#         # Wait until all threads finish computing
#         cuda.syncthreads()

#     E[x, y] = tmp

# def ET(H, EVV, betas, EVU):
#     with tf.device(device):
#         return H[:, tnp.newaxis] * (EVV - betas[:, tnp.newaxis] * EVU)


    # M['H'][s, t, r, :, np.newaxis] * (
    #     M['EVV'][s, curr_t, r] - M["betas"][s, r, :, np.newaxis] * M['EVU'][s, curr_t, r])
def process_layer(cfg, M, t, s, r, W_rec, betas):
    """ Process a single layer of the model at a single time step."""
    # If not tracking state, the time dimensions of synaptic variables have
    # length 1 and we overwrite those previous time steps at index 0.
    start = time.time()
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

    M["Z_in"][s, t, r] = np.concatenate((M["Z_prev"][s, t, r],
                                         M['Z'][s, t-1, r]))

    # Revert eprop functions EVV and V if using traub trick!
    assert(not cfg["traub_trick"])

    # EVU

    M['EVU'][s, curr_t, r] = (M['H'][s, t-1, r, :, np.newaxis] * M['EVV'][s, prev_t, r]
                              + (cfg["rho"] - M['H'][s, t-1, r] * betas)[:, np.newaxis] * M['EVU'][s, prev_t, r])

    # EVV
    M['EVV'][s, curr_t, r] = (cfg["alpha"] * M['EVV'][s, prev_t, r]
                              + M["Z_in"][s, t, r])

    # I
    M['I_in'][s, t, r] = np.dot(W_rec[:, :cfg["N_R"]], M["Z_prev"][s, t, r])
    M['I_rec'][s, t, r] = np.dot(W_rec[:, cfg["N_R"]:], M['Z'][s, t-1, r])

    M['I'][s, t, r] = M['I_in'][s, t, r] + M['I_rec'][s, t, r]

    # V
    M['V'][s, t, r] =  (cfg["alpha"] * M['V'][s, t-1, r]
                        + M['I'][s, t, r]
                        - M['Z'][s, t-1, r] * (
                            M['A'][s, t-1, r] if cfg["v_fix"] else cfg["thr"]))

    # a
    M['a'][s, t, r] = cfg['rho'] * M['a'][s, t-1, r] + M['Z'][s, t-1, r]
    M['A'][s, t, r] = cfg['thr'] + betas * M['a'][s, t, r]

    # Update the spikes Z of the neurons.
    M['Z'][s, t, r] = np.where(
        np.logical_and(t - M['TZ'][s, r] >= cfg["dt_refr"],
                       M['V'][s, t, r] >= M['A'][s, t, r]),
        1,
        0)

    # TZ is time of latest spike
    M['TZ'][s, r, M['Z'][s, t, r]==1] = t

    # Pseudoderivative H
    M['H'][s, t, r] = ((1 / cfg["thr"] if not cfg["traub_trick"] else 1)
                       * cfg["gamma"] * np.clip(
                            a=1 - (abs((M['V'][s, t, r] - M['A'][s, t, r]
                                       / cfg["thr"]))),
                            a_min=0,
                            a_max=None))

    # ET
    M['ET'][s, curr_t, r] = M['H'][s, t, r, :, np.newaxis] * (
        M['EVV'][s, curr_t, r] - betas[:, np.newaxis] * M['EVU'][s, curr_t, r])

    # Input layer always "spikes" (with nonbinary spike values Z=X).
    # TZ_prev = M['TZ'][s, r-1] if r > 0 else \
    #     np.ones(shape=(M['TZ'][s, r].shape)) * t

    # M["TZ_in"][s, r] = np.concatenate((TZ_prev, M['TZ'][s, r]-1))

    # M['Z_inbar'][s, t] = (cfg["alpha"] * (M['Z_inbar'][s, t-1] if t > 0 else 0)) + M['Z_in'][s, t]

    M['ETbar'][s, curr_t, r] = cfg["kappa"] * M['ETbar'][s, prev_t, r] + M['ET'][s, curr_t, r]
    M['Zbar'][s, t, r] = cfg["kappa"] * (M['Zbar'][s, t-1, r]) + M['Z'][s, t, r]
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


def load_data(cfg):

    inps = {}
    tars = {}
    rng = np.random.default_rng(seed=cfg["seed"])
    for tvt_type in cfg['n_examples'].keys():
        inps[tvt_type] = np.load(f"{cfg['wavs_fname']}_{tvt_type}_{cfg['task']}.npy")

    mintrain = np.amin(inps['train'], axis=(0, 1))
    maxtrain = np.ptp(inps['train'], axis=(0, 1))

    for tvt_type in cfg['n_examples'].keys():
        # Normalize [0, 1]

        inps[tvt_type] = np.where(inps[tvt_type] != -1, inps[tvt_type] - mintrain, -1)
        inps[tvt_type] = np.where(inps[tvt_type] != -1, inps[tvt_type] / maxtrain, -1)

        tars[tvt_type] = np.load(f"{cfg['phns_fname']}_{tvt_type}_{cfg['task']}.npy")

        shuf_idxs = np.arange(inps[tvt_type].shape[0])
        rng.shuffle(shuf_idxs)
        inps[tvt_type] = inps[tvt_type][shuf_idxs]
        tars[tvt_type] = tars[tvt_type][shuf_idxs]


    return inps, tars

def interpolate_inputs(cfg, inp, tar, stretch):
    if inp.ndim == 2:
        inp = inp.T
        tar = tar.T
    else:
        inp = inp.transpose((0, 2, 1))
        tar = tar.transpose((0, 2, 1))

    itp_inp = interp1d(np.arange(inp.shape[-1]),
                       inp,
                       kind=cfg["Interpolation"])

    itp_tar = interp1d(np.arange(tar.shape[-1]),
                       tar,
                       kind='nearest')

    inp = itp_inp(np.linspace(0, inp.shape[-1]-1, int(inp.shape[-1]*stretch)))
    tar = itp_tar(np.linspace(0, tar.shape[-1]-1, int(tar.shape[-1]*stretch)))

    if inp.ndim == 2:
        inp = inp.T
        tar = tar.T
    else:
        inp = inp.transpose((0, 2, 1))
        tar = tar.transpose((0, 2, 1))

    return inp, tar

    # this_inps = np.repeat(this_inps, cfg["Repeats"], axis=0)
    # this_tars = np.repeat(this_tars, cfg["Repeats"], axis=0)
