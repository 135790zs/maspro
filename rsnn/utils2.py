
import datetime
import numpy as np
import os
import json
from scipy.interpolate import interp1d
import time
import opt_einsum
import torch
from config2 import cfg

if cfg["cuda"]:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

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

    for var in ['z', 'I', 'I_in', 'I_rec', 'h', 'v', 'a', 'zbar', 'l', 'l_std', 'l_fr']:
        M[var] = torch.zeros(size=nrn_shape, dtype=torch.double)

    for var in ['vv', 'va', 'etbar']:
        M[var] = torch.zeros(size=syn_shape, dtype=torch.double)

    M['tz'] = torch.ones(size=(cfg["n_directions"],
                               batch_size,
                               cfg["N_Rec"],
                               cfg["N_R"]), dtype=torch.int) * -cfg["dt_refr"]

    for var in ['y', 'p', 'd', 'pm']:
        M[var] = torch.zeros(size=out_shape, dtype=torch.double)
    M['ysub'] = torch.zeros(size=(cfg["n_directions"],
                                batch_size,
                                n_steps,
                                tar_size), dtype=torch.double)

    M["ce"] = torch.zeros(size=(batch_size, n_steps,))
    M["correct"] = torch.zeros(size=(batch_size, n_steps,))

    return M


def initialize_gradients(cfg, tar_size, batch_size):
    G = {}
    G["W"] = torch.zeros(size=(cfg["n_directions"],
                             batch_size,
                             cfg["N_Rec"],
                             cfg["N_R"],
                             cfg["N_R"] * 2,))
    G["out"] = torch.zeros(size=(cfg["n_directions"],
                               batch_size,
                               tar_size,
                               cfg["N_R"]))

    G["bias"] = torch.zeros(size=(batch_size,
                                tar_size,))
    return G


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
                              cfg["N_R"] * 2,))

    # Bellec: rd.randn(n_rec, n_rec) / np.sqrt(n_rec)
    W['W'][:, :, :, cfg["N_R"]:] /= np.sqrt(cfg["N_R"])

    # Bellec: rd.randn(n_in, n_rec) / np.sqrt(n_in)
    for r in range(cfg["N_Rec"]):
        W['W'][:, 0, :, :cfg["N_R"]] /= np.sqrt(inp_size)
        if r > 0:
            W['W'][:, r, :, :cfg["N_R"]] /= np.sqrt(cfg["N_R"])
    W['W'][:, 0, :, inp_size:cfg["N_R"]] = 0
    W["out"] = rng.normal(size=(cfg["n_directions"],
                                tar_size,
                                cfg["N_R"])) / np.sqrt(tar_size)

    W["bias"] = np.zeros(shape=(tar_size,))

    B_shape = (cfg["n_directions"],
               cfg["N_Rec"],
               cfg["N_R"],
               tar_size,)

    if cfg["eprop_type"] == "random":  # Gaussian, variance of 1
        W["B"] = rng.normal(size=B_shape, scale=1)

    elif cfg["eprop_type"] == "adaptive":  # Gaussian, variance of 1/N
        W["B"] = rng.normal(size=B_shape, scale=np.sqrt(1 / cfg["N_R"]))

    elif cfg["eprop_type"] == "symmetric":
        W["B"] = rng.random(size=B_shape) * 2 - 1
        for r in range(cfg["N_Rec"]):
            for s in range(cfg["n_directions"]):
                W["B"][s, r] = W["out"][s].T

    # Drop all self-looping weights. A neuron cannot be connected with itself.
    for r in range(cfg["N_Rec"]):
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W['W'][s, r, :, cfg["N_R"]:], 0)

    W['W'][..., cfg["N_R"]:] = np.where(rng.random(W['W'][..., cfg["N_R"]:].shape) < cfg["dropout"],
                         0,
                         W['W'][..., cfg["N_R"]:])

    for wtype in ["W", "out", "bias", "B"]:
        W[wtype] = torch.tensor(W[wtype])

    return W


def update_weights(W, G, adamvars, e, cfg):
    adamvars['it'] += 1

    for wtype in G.keys():
        if not cfg[f"train_{wtype}"]:
            continue

        adamvars[f"m{wtype}"] = (cfg["adam_beta1"] * adamvars[f"m{wtype}"]
                                 + (1 - cfg["adam_beta1"]) * G[wtype])
        adamvars[f"v{wtype}"] = (cfg["adam_beta2"] * adamvars[f"v{wtype}"]
                                 + (1 - cfg["adam_beta2"]) * G[wtype] ** 2)

        m = adamvars[f"m{wtype}"] / (1 - cfg["adam_beta1"] ** adamvars['it'])
        v = adamvars[f"v{wtype}"] / (1 - cfg["adam_beta2"] ** adamvars['it'])

        dw = -cfg[f"eta_{wtype}"] * (m / (torch.sqrt(v) + cfg["adam_eps"]))
        W[wtype] += dw

        if wtype == 'out' and cfg['eprop_type'] == 'adaptive':
            W['B'] += dw

            W['out'] -= cfg["weight_decay"] * W['out']
            W['B'] -= cfg["weight_decay"] * W['B']

    if cfg["eprop_type"] == 'symmetric':
        for s in range(cfg['n_directions']):
            for r in range(cfg['N_Rec']):
                W['B'][s, r] = W['out'][s].T

    return W, adamvars


def eprop(cfg, X, T, n_steps, betas, W):


    # Trim input down to layer size.
    X = X[:, :, :cfg["N_R"]]
    inp_size = X.shape[-1]

    X = np.pad(X, ((0,0), (0,0), (0, cfg["N_R"] - inp_size)))

    X = torch.tensor(X)
    T = torch.tensor(T)
    n_steps = torch.tensor(n_steps)

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
        M['x'] = torch.cat((M['x'], M['x']))
        for b in range(M['x'].shape[1]):
            M['x'][1, b, :n_steps[b]] = torch.fliplr(M['x'][0, b, :n_steps[b]])

    for t in torch.arange(0, max(n_steps), dtype=torch.int):
        start = time.time()
        prev_t, curr_t = conn_t_idxs(track_synapse=cfg['Track_synapse'], t=t)

        is_valid = torch.any(M['x'][:, :, t] != -1, axis=2)

        for r in range(cfg["N_Rec"]):  # TODO: Can overwrite r instead of appending, except Z
            Z_prev = M['z'][:, :, t, r-1] if r else M['x'][:, :, t]
            Z_in = torch.cat((Z_prev, M['z'][:, :, t-1, r]), axis=2)

            M['va'][:, :, curr_t, r] = (opt_einsum.contract("sbj, sbji -> sbji",
                                                  M['h'][:, :, t-1, r],
                                                  M['vv'][:, :, prev_t, r],
                                                  backend='torch')
                                        + opt_einsum.contract("sbj, sbji -> sbji",
                                                    cfg["rho"] - opt_einsum.contract(
                                                        "sbj, sj -> sbj",
                                                        M['h'][:, :, t-1, r],
                                                        betas[:, r]),
                                                    M['va'][:, :, prev_t, r],
                                                    backend='torch'))

            # TIMESINK
            M['vv'][:, :, curr_t, r] = opt_einsum.contract("sbji, sbi -> sbji",
                                                    cfg["alpha"] * M['vv'][:, :, prev_t, r],
                                                    Z_in,
                                                    backend='torch')  # TODO: not sure!


            # TIMESINK
            M['vv'][:, :, curr_t, r] = cfg["alpha"] * M['vv'][:, :, prev_t, r] + Z_in[:, :, np.newaxis, :]
            M['I_in'][:, :, t, r] = opt_einsum.contract("sji, sbi -> sbj",
                                        W['W'][:, r, :, :cfg["N_R"]],
                                        Z_prev,
                                        backend='torch')
            M['I_rec'][:, :, t, r] = opt_einsum.contract("sji, sbi -> sbj",
                                        W['W'][:, r, :, cfg["N_R"]:],
                                        M['z'][:, :, t-1, r],
                                        backend='torch')
            M['I'][:, :, t, r] = M['I_in'][:, :, t, r] + M['I_rec'][:, :, t, r]

            M['a'][:, :, t, r] = (cfg["rho"] * M['a'][:, :, t-1, r]
                                  + M['z'][:, :, t-1, r])
            A = cfg["thr"] + opt_einsum.contract("sj, sbj -> sbj",
                                                 betas[:, r],
                                                 M['a'][:, :, t, r])

            M['v'][:, :, t, r] = (cfg["alpha"] * M['v'][:, :, t-1, r]
                                  + M['I'][:, :, t, r]
                                  - M['z'][:, :, t-1, r] * (A if cfg["v_fix"] else cfg["thr"]))

            M['z'][:, :, t, r] = torch.where(
                torch.logical_and(t - M['tz'][:, :, r] >= cfg["dt_refr"],
                               M['v'][:, :, t, r] >= A),
                1,
                0)
            M['tz'][:, :, r] = torch.where(M['z'][:, :, t, r] != 0, t, M['tz'][:, :, r])

            M['h'][:, :, t, r] = ((1 / (A if cfg["v_fix"] else cfg["thr"])) * cfg["gamma"] * torch.clip(
            # M['h'][:, :, t, r] = (cfg["gamma"] * torch.clip(
                1 - (abs((M['v'][:, :, t, r] - A
                           / (A if cfg["v_fix"] else cfg["thr"])))),
                0,
                None))

            # TIMESINK
            ET = (opt_einsum.contract("sbj, sbji -> sbji",
                                      M['h'][:, :, t, r],
                                      M['vv'][:, :, curr_t, r],
                                      backend='torch')
                  - opt_einsum.contract("sj, sbji -> sbji",
                                        betas[:, r],
                                        M['va'][:, :, curr_t, r],
                                        backend='torch'))

            M['etbar'][:, :, curr_t, r] = cfg["kappa"] * M['etbar'][:, :, prev_t, r] + ET
            M['zbar'][:, :, t, r] = cfg["kappa"] * M['zbar'][:, :, t-1, r] + M['z'][:, :, t, r]

        # TODO: Can vectorize (over t) everything below here
        M['ysub'][:, :, t] = opt_einsum.contract("skj, sbj -> sbk", # Checked correct
                                           W['out'],
                                           M['z'][:, :, t, -1],
                                           backend='torch')

        M['ysub'][:, :, t] += cfg["kappa"] * M['ysub'][:, :, t-1]
        M['y'][:, t] = torch.sum(M['ysub'][:, :, t], axis=0)
        M['y'][:, t] += W['bias']

        M['p'][:, t] = torch.exp(M['y'][:, t] - torch.amax(M['y'][:, t], axis=1)[:, np.newaxis])
        M['p'][:, t] = M['p'][:, t] / torch.sum(M['p'][:, t], axis=1)[:, np.newaxis]


        M['d'][:, t] = M['p'][:, t] - T[:, t]

        M['l_std'][:, :, t] = opt_einsum.contract("srjk, bk -> sbrj",
                      W['B'],
                      M['d'][:, t],
                      backend='torch')

        # Dividing by number of total steps isn't really online!
        M['l_fr'][:, :, t] = (cfg["FR_reg"] / n_steps[np.newaxis, :, np.newaxis, np.newaxis]
              * (torch.mean(M['z'][:, :, :t+1], axis=2) - cfg["FR_target"]))
        M['l'][:, :, t] = (M['l_std'][:, :, t] + M['l_fr'][:, :, t])

        # TIMESINK:
        G['W'] += opt_einsum.contract("sbrj, sbrji -> sbrji",
                                      M['l_std'][:, :, t],
                                      M['etbar'][:, :, curr_t],
                                      backend='torch') * is_valid[..., None, None, None]
        G['W'] += opt_einsum.contract("sbrj, sbrji -> sbrji",
                                      M['l_fr'][:, :, t],
                                      torch.ones_like(M['etbar'][:, :, curr_t]),
                                      backend='torch') * is_valid[..., None, None, None]

        G['out'] += opt_einsum.contract("bk, sbj -> sbkj",
                                 M['d'][:, t],
                                 M['zbar'][:, :, t, -1],
                                 backend='torch') * is_valid[..., None, None]

        G['bias'] += torch.sum(M['d'][:, t] * is_valid[..., None], axis=0)

    a = torch.arange(M['p'].shape[1])
    for b in range(M['p'].shape[0]):
        M['pm'][b, a, M['p'][b].argmax(axis=1)] = 1

    M['correct'] = (M['pm'] == M['t']).all(axis=2)

    M['ce'] = -torch.sum(M['t'] * torch.log(1e-30 + M['p']), axis=2)

    # Batch mean
    G['W'] = torch.mean(G['W'], axis=1)
    G['out'] = torch.mean(G['out'], axis=1)
    G['bias'] = torch.mean(G['bias'], axis=0)

    # Don't update dead weights
    G['W'][W['W']==0] = 0

    # L2 regularization
    for wtype in ['W', 'out', 'bias']:
        G[wtype] += cfg["L2_reg"] * torch.linalg.norm(W[wtype].flatten())

    return G, M


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
        'mW': torch.zeros(size=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'vW': torch.zeros(size=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'mout': torch.zeros(size=(cfg["n_directions"], tar_size, cfg["N_R"],)),
        'vout': torch.zeros(size=(cfg["n_directions"], tar_size, cfg["N_R"],)),
        'mbias': torch.zeros(size=(tar_size,)),
        'vbias': torch.zeros(size=(tar_size,)),
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
    betas = torch.tensor(betas)
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
        wf = w.flatten()
        if wtype != 'bias':
            ssize = min(wf.shape[0], sample_size)
            idxs = rng.choice(ssize, size=ssize, replace=False)[:wf.shape[0]]
        else:
            idxs = np.arange(wf.shape[0])

        W_log[f'{wtype}_idxs'] = idxs
        # W_log[wtype] = w.flatten()[W_log[f'{wtype}_idxs']][:, np.newaxis]
        W_log[wtype] = np.empty(shape=(idxs.size, 0))


    return W_log


def update_W_log(W_log, Mt, Mv, W):
    # weights sample
    for wtype, w in W.items():
        wcpu = w.cpu().numpy()
        W_log[wtype] = np.append(W_log[wtype], wcpu.flatten()[W_log[f'{wtype}_idxs']][:, np.newaxis], axis=1)

    for tv_type, M in (('train', Mt), ('val', Mv)):
        if M is None:  # Mv skipped

            W_log['Cross-entropy'][tv_type].append(-1)
            W_log['Mean Hz'][tv_type].append(-1)
            W_log['Percentage wrong'][tv_type].append(-1)
            continue

        X = M['x'].cpu().numpy()
        bsize = X.shape[1]
        ces = np.zeros(shape=(bsize))
        hz = np.zeros(shape=(bsize))
        pwrong = np.zeros(shape=(bsize))

        for b in range(bsize):
            arr = X[0, b]
            while np.any(arr[-1] == -1):
                arr = arr[:-1]

            ces[b] = np.mean(M['ce'].cpu().numpy()[b, :arr.shape[0]])
            pwrong[b] = 100-100*np.mean(M['correct'].cpu().numpy()[b, :arr.shape[0]])
            hz[b] = 1000*np.mean(np.mean(M['z'][:, b, :arr.shape[0]].cpu().numpy(), axis=2))

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
