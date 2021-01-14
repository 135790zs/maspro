
import datetime
import numpy as np
import os
import json
from scipy.interpolate import interp1d


def initialize_model(cfg, inp_size, tar_size, batch_size, n_steps):
    M = {}

    len_syn_time = n_steps if cfg["Track_synapse"] else 1

    nrn_shape = (cfg["n_directions"],
                 batch_size,
                 n_steps,
                 cfg["N_Rec"],
                 cfg["N_R"])
    syn_shape = (cfg["n_directions"],
                 batch_size,
                 len_syn_time,
                 cfg["N_Rec"],
                 cfg["N_R"],
                 cfg["N_R"] * 2)
    out_shape = (batch_size,
                 n_steps,
                 tar_size)

    for var in ['z', 'I', 'h', 'v', 'a', 'zbar', 'l']:
        M[var] = np.zeros(shape=nrn_shape)

    for var in ['vv', 'va', 'etbar']:
        M[var] = np.zeros(shape=syn_shape)

    M['tz'] = np.ones(shape=(cfg["n_directions"],
                             batch_size,
                             cfg["N_Rec"],
                             cfg["N_R"])) * -cfg["dt_refr"]

    for var in ['y', 'p', 'd', 'pm']:
        M[var] = np.zeros(shape=out_shape)
    M['ysub'] = np.zeros(shape=(cfg["n_directions"],
                                batch_size,
                                n_steps,
                                tar_size))

    M["ce"] = np.zeros(shape=(batch_size, n_steps,))
    M["correct"] = np.zeros(shape=(batch_size, n_steps,))

    return M


def initialize_gradients(cfg, tar_size, batch_size):
    G = {}
    G["W"] = np.zeros(shape=(cfg["n_directions"],
                             batch_size,
                             cfg["N_Rec"],
                             cfg["N_R"],
                             cfg["N_R"] * 2,))
    G["out"] = np.zeros(shape=(cfg["n_directions"],
                               batch_size,
                               tar_size,
                               cfg["N_R"]))

    G["bias"] = np.zeros(shape=(batch_size,
                                tar_size,))
    return G


def eprop(cfg, X, T, n_steps, betas, W):
    inp_size = X.shape[-1]
    M = initialize_model(cfg=cfg,
                         inp_size=inp_size,
                         tar_size=T.shape[-1],
                         batch_size=X.shape[0],
                         n_steps=max(n_steps))

    G = initialize_gradients(cfg=cfg,
                             tar_size=T.shape[-1],
                             batch_size=X.shape[0])

    M['x'] = X[np.newaxis, :]  # Insert subnetwork dimension
    M['t'] = T

    if cfg["n_directions"] == 2:
        # TODO: Flip -1's accordingly
        M['x'] = np.concatenate((X, np.flip(X, axis=0)))

    # Trim input down to layer size.
    M['x'] = M['x'][:, :, :, :cfg["N_R"]]

    for t in range(max(n_steps)):
        print(t)
        prev_t, curr_t = conn_t_idxs(track_synapse=cfg['Track_synapse'], t=t)

        is_valid = np.any(X[0, :] != -1)

        for r in range(cfg["N_Rec"]):  # TODO: Can overwrite r instead of appending, except Z
            Z_prev = Z[:, :, t, r-1] if r else \
                np.pad(M['x'][:, :, t],
                       ((0, 0), (0, 0), (0, max(0, cfg["N_R"] - inp_size))))

            Z_in = np.concatenate((Z_prev, M['z'][:, :, t-1, r]), axis=2)

            M['va'][:, :, curr_t, r] = (np.einsum("sbj, sbji -> sbji",
                                                  M['h'][:, :, t-1, r],
                                                  M['vv'][:, :, prev_t, r])
                                        + np.einsum("sbj, sbji -> sbji",
                                                    cfg["rho"] - M['h'][:, :, t-1, r] * betas,
                                                    M['va'][:, :, prev_t, r]))

            M['vv'][:, :, curr_t, r] = np.einsum("sbji, sbi -> sbji",
                                                 cfg["alpha"] * M['vv'][:, :, prev_t, r],
                                                 Z_in)  # TODO: not sure!

            M['vv'][:, :, curr_t, r] = cfg["alpha"] * M['vv'][:, :, prev_t, r] + Z_in[:, :, np.newaxis, :]
            M['I'][:, :, t, r] = np.einsum("sji, sbi -> sbj",
                                        W['W'][:, r],
                                        Z_in)
            M['v'][:, :, t, r] = (cfg["alpha"] * M['v'][:, :, t-1, r]
                                  + M['I'][:, :, t, r]
                                  - M['z'][:, :, t-1, r] * cfg["thr"])  # TODO: vfix

            M['a'][:, :, t, r] = (cfg["rho"] * M['a'][:, :, t-1, r]
                                  + M['z'][:, :, t-1, r])
            A = cfg["thr"] + betas * M['a'][:, :, t, r]

            M['z'][:, :, t, r] = np.where(
                np.logical_and(t - M['tz'][:, :, r] >= cfg["dt_refr"],
                               M['v'][:, :, t, r] >= A),
                1,
                0)

            M['tz'][:, :, r] = np.where(M['z'][:, :, t, r], t, M['tz'][:, :, r])

            M['h'][:, :, t, r] = ((1 / cfg["thr"]) * cfg["gamma"] * np.clip(
                a=1 - (abs((M['v'][:, :, t, r] - A
                           / cfg["thr"]))),
                a_min=0,
                a_max=None))
            ET = (np.einsum("sbj, sbji -> sbji", M['h'][:, :, t, r], M['vv'][:, :, curr_t, r])
                  - np.einsum("srj, sbji -> sbji", betas, M['va'][:, :, curr_t, r]))

            M['etbar'][:, :, curr_t, r] = cfg["kappa"] * M['etbar'][:, :, prev_t, r] + ET
            M['zbar'][:, :, t, r] = cfg["kappa"] * M['zbar'][:, :, t-1, r] + M['z'][:, :, t, r]

        # TODO: Can vectorize (over t) everything below here
        M['ysub'][:, :, t] = (cfg["kappa"] * M['ysub'][:, :, t-1])

        M['ysub'][:, :, t] += np.einsum("skj, sbj -> sbk",  # TODO: not sure
                                        W['out'],
                                        M['z'][:, :, t, -1])
        M['y'][:, t] = np.sum(M['ysub'][:, :, t], axis=0)

        M['y'][:, t] += W['bias']
        M['p'][:, t] = np.exp(M['y'][:, t] - np.amax(M['y'][:, t], axis=1)[:, np.newaxis])
        M['p'][:, t] = M['p'][:, t] / np.sum(M['p'][:, t], axis=1)[:, np.newaxis]

        M['ce'][:, t] = -np.sum(M['t'][:, t] * np.log(1e-30 + M['p'][:, t]))

        for b in np.arange(M['p'].shape[0]):
            M['pm'][b, t, M['p'][b, t].argmax()] = 1
            M['correct'][b, t] = int((M['pm'][b, t] == M['t'][b, t]).all())


        M['d'][:, t] = M['p'][:, t] - T[:, t]
        M['l'][:, :, t] = np.einsum("srjk, bk -> sbrj",
                      W['B'],
                      M['d'][:, t])

        # TODO: Think of a way to precisely get mean over time, not overweigh initials
        M['l'][:, :, t] += (cfg["FR_reg"] / n_steps[np.newaxis, :, np.newaxis, np.newaxis]
              * (M['z'][:, :, t] - cfg["FR_target"]))

        G['W'] += np.einsum("sbrj, sbrji -> sbrji", M['l'][:, :, t], M['etbar'][:, :, curr_t]) * is_valid
        G['out'] += np.einsum("bk, sbrj -> sbkj", M['d'][:, t], M['zbar'][:, :, curr_t]) * is_valid
        G['bias'] += np.sum(M['d'][:, t], axis=0) * is_valid


    # Time mean
    G['W'] /= n_steps[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    G['out'] /= n_steps[np.newaxis, :, np.newaxis, np.newaxis]
    G['bias'] /= n_steps[:, np.newaxis]

    # Batch mean
    G['W'] = np.mean(G['W'], axis=1)
    G['out'] = np.mean(G['out'], axis=1)
    G['bias'] = np.mean(G['bias'], axis=0)

    # Don't update dead weights
    G['W'][W['W']==0] = 0

    # L2 regularization
    for wtype in ['W', 'out', 'bias']:
        G[wtype] += cfg["L2_reg"] * np.linalg.norm(W[wtype].flatten())

    return G, M


def update_weights(W, G, adamvars, e, cfg):
    adamvars['it'] += 1

    for wtype in G.keys():

        adamvars[f"m{wtype}"] = (cfg["adam_beta1"] * adamvars[f"m{wtype}"]
                                 + (1 - cfg["adam_beta1"]) * G[wtype])
        adamvars[f"v{wtype}"] = (cfg["adam_beta2"] * adamvars[f"v{wtype}"]
                                 + (1 - cfg["adam_beta2"]) * G[wtype] ** 2)

        m = adamvars[f"m{wtype}"] / (1 - cfg["adam_beta1"] ** adamvars['it'])
        v = adamvars[f"v{wtype}"] / (1 - cfg["adam_beta2"] ** adamvars['it'])

        W[wtype] -= cfg["eta"] * (m / (np.sqrt(v) + cfg["adam_eps"]))

    return W, adamvars


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


def init_adam(cfg, tar_size):
    return {
        'mW': np.zeros(shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'vW': np.zeros(shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'mout': np.zeros(shape=(cfg["n_directions"], tar_size, cfg["N_R"],)),
        'vout': np.zeros(shape=(cfg["n_directions"], tar_size, cfg["N_R"],)),
        'mbias': np.zeros(shape=(tar_size,)),
        'vbias': np.zeros(shape=(tar_size,)),
        'it': 0
    }


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


def conn_t_idxs(t, track_synapse):  # TODO: Rework this and previous function to same idea
    if track_synapse:
        return t-1, t
    return 0, 0


def initialize_W_log(cfg, W, sample_size=100):
    W_log = {}

    rng = np.random.default_rng(seed=cfg['seed'])
    W_log['Cross-entropy'] = {'train': [], 'val': []}
    W_log['Percentage wrong'] = {'train': [], 'val': []}
    W_log['Mean Hz'] = {'train': [], 'val': []}

    for wtype, w in W.items():
        if wtype != 'bias':
            ssize = min(w.size, sample_size)
            idxs = rng.choice(ssize, size=ssize, replace=False)[:w.size]
        else:
            idxs = np.arange(w.size)

        W_log[f'{wtype}_idxs'] = idxs
        W_log[wtype] = w.flatten()[W_log[f'{wtype}_idxs']][:, np.newaxis]


    return W_log


def update_W_log(W_log, Mt, Mv, W):
    # weights sample
    for wtype, w in W.items():
        W_log[wtype] = np.append(W_log[wtype], w.flatten()[W_log[f'{wtype}_idxs']][:, np.newaxis], axis=1)

    for tv_type, M in (('train', Mt), ('val', Mv)):
        if M is None:  # Mv skipped

            W_log['Cross-entropy'][tv_type].append(-1)
            W_log['Mean Hz'][tv_type].append(-1)
            W_log['Percentage wrong'][tv_type].append(-1)
            continue

        bsize = M['x'].shape[1]
        ces = np.zeros(shape=(bsize))
        hz = np.zeros(shape=(bsize))
        pwrong = np.zeros(shape=(bsize))

        for b in range(bsize):
            arr = M['x'][0, b]
            while np.all(arr[-1] == -1):
                arr = arr[:-1]

            ces[b] = np.mean(M['ce'][b, :arr.shape[0]])
            pwrong[b] = 100-100*np.mean(M['correct'][b, :arr.shape[0]])
            hz[b] = 1000*np.mean(np.mean(M['z'][:, b, :arr.shape[0]], axis=2))

        W_log['Cross-entropy'][tv_type].append(np.mean(ces))
        W_log['Mean Hz'][tv_type].append(np.mean(hz))
        W_log['Percentage wrong'][tv_type].append(np.mean(pwrong))

    return W_log


def trim_samples(X, T=None):
    # Crop -1's
    while np.all(X[:, -1] == -1):
        X = X[:, :-1]
    if T is not None:
        T = T[:, :X.shape[1]]
        return X, T
    return X


def count_lengths(X):
    n_steps = np.zeros(shape=(X.shape[0]), dtype=np.int32)
    for idx, arr in enumerate(X):
        while np.any(arr[-1] == -1):
            arr = arr[:-1]
        n_steps[idx] = arr.shape[0]
    return n_steps


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


def sample_mbatches(cfg, n_train):
    ret = []
    samples = np.arange(n_train)

    rng = np.random.default_rng(seed=cfg["seed"])
    rng.shuffle(samples)

    while samples.shape[0]:
        ret.append(samples[:cfg["batch_size_train"]])
        samples = samples[cfg["batch_size_train"]:]

    return ret


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

    """ See Supplementary information for: Long short-term
    memory and learning-to-learn in networks of spiking
    neurons """
    W["W"] = rng.normal(size=(cfg["n_directions"],
                              cfg["N_Rec"],
                              cfg["N_R"],
                              cfg["N_R"] * 2,)) / np.sqrt(cfg["N_R"] + inp_size)
    W["out"] = rng.normal(size=(cfg["n_directions"],
                                tar_size,
                                cfg["N_R"])) / np.sqrt(cfg["N_R"])

    W["bias"] = np.zeros(shape=(tar_size,))

    B_shape = (cfg["n_directions"],
               cfg["N_Rec"],
               cfg["N_R"],
               tar_size,)

    if cfg["eprop_type"] == "random":  # Gaussian, variance of 1
        W["B"] = rng.normal(size=B_shape, scale=1)

    elif cfg["eprop_type"] == "global":  # Gaussian, variance of 1
        W["B"] = np.ones(shape=B_shape) * (1 / np.sqrt(cfg["N_R"] * cfg["N_Rec"]))

    elif cfg["eprop_type"] == "adaptive":  # Gaussian, variance of 1/N
        W["B"] = rng.normal(size=B_shape, scale=np.sqrt(1 / cfg["N_R"]))

    else:  # Symmetric: Uniform [0, 1]. Irrelevant, as it'll be overwritten
        W["B"] = rng.random(size=B_shape) * 2 - 1
        for r in range(cfg["N_Rec"]):
            for s in range(cfg["n_directions"]):
                W["B"][s, r] = W["out"][s].T

    # Drop all self-looping weights. A neuron cannot be connected with itself.
    for r in range(cfg["N_Rec"]):
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W['W'][s, r, :, cfg["N_R"]:], 0)

    return W


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


def get_log_id():
    log_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    id_append = ""
    count = 1
    while os.path.isdir(f"../log/{log_id}{id_append}"):
        id_append = f"_{count}"
        count += 1
    return log_id + id_append
