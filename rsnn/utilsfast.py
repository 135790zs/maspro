import datetime
import numpy as np
import os
import json
from scipy.interpolate import interp1d
import time
import torch
from config import cfg

if cfg["cuda"]:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def initialize_model(cfg, inp_size, tar_size, batch_size, n_steps):
    M = {}

    nrn_shape = (batch_size,
                 cfg["N_Rec"],
                 cfg["N_R"])
    nrn_shape_timed = (batch_size,
                       n_steps,
                       cfg["N_Rec"],
                       cfg["N_R"])
    nrn_shape_db = (batch_size,
                    cfg["N_Rec"],
                    cfg["N_R"]*2)
    syn_shape = (batch_size,
                 cfg["N_Rec"],
                 cfg["N_R"],
                 cfg["N_R"] * 2)
    syn_shape_nobatch = (cfg["N_Rec"],
                         cfg["N_R"],
                         cfg["N_R"] * 2)

    out_shape = (batch_size,
                 tar_size)


    for var in ['zs', 'h', 'v', 'a']:
        M[var] = torch.zeros(size=nrn_shape)

    for var in ['z']:
        M[var] = torch.zeros(size=nrn_shape_timed)

    nrn_shape_single = (batch_size,
                        cfg["N_R"])

    for var in ['zbar']:
        M[var] = torch.zeros(size=nrn_shape_single)

    for var in ['vv', 'va', 'etbar']:
        M[var] = torch.zeros(size=syn_shape)

    for var in ['GW']:
        M[var] = torch.zeros(size=syn_shape_nobatch)

    M['tz'] = torch.ones(size=nrn_shape, dtype=torch.int) * -cfg["dt_refr"]

    for var in ['y']:
        M[var] = torch.zeros(size=out_shape)

    out_shape_long = (batch_size,
                      n_steps,
                      tar_size)

    for var in ['p','pm']:
        M[var] = torch.zeros(size=out_shape_long)

    M["ce"] = torch.zeros(size=(batch_size, n_steps,))
    M["correct"] = torch.zeros(size=(batch_size, n_steps,))

    M["Gout"] = torch.zeros(size=(cfg["N_R"],
                                  tar_size))
    M["Gbias"] = torch.zeros(size=(tar_size,))

    return M


def initialize_gradients(cfg, tar_size, batch_size):
    G = {}
    G["W"] = torch.zeros(size=(batch_size,
                               cfg["N_Rec"],
                               cfg["N_R"],
                               cfg["N_R"] * 2,))
    G["out"] = torch.zeros(size=(batch_size,
                                 cfg["N_R"],
                                 tar_size,))
    G["bias"] = torch.zeros(size=(batch_size,
                                  tar_size,))
    return G


def initialize_weights(cfg, inp_size, tar_size):
    """ Initializes the variables used to train a network's weights.

    The difference with the Model is that the model re-initializes for
    every new time series, while the Weights in this function are persistent
    over many epochs. They are also stored in an array such that the
    evolution of the weights over the number of epochs can be inspected.

    out:   Output weights
    bias:  Bias
    W:     Network weights
    B:     Broadcast weights
    """

    W = {}
    rng = np.random.default_rng(seed=cfg["seed"])

    W["W"] = rng.normal(size=(cfg["N_Rec"],
                              cfg["N_R"],
                              cfg["N_R"] * 2,))
    # Bellec: rd.randn(n_in, n_rec) / np.sqrt(n_in)
    W['W'][0, :, inp_size:cfg['N_R']] = 0  # From padded channels
    W['W'][0, :, :inp_size] /= np.sqrt(inp_size)

    for r in range(cfg["N_Rec"]):
        if r > 0:
            W['W'][r, :, :cfg['N_R']] /= np.sqrt(cfg["N_R"])

    W['W'][:, :, cfg["N_R"]:] /= np.sqrt(cfg["N_R"])


    W["out"] = rng.normal(size=(cfg["N_R"],
                                tar_size)) / np.sqrt(tar_size)

    W["bias"] = np.zeros(shape=(tar_size,))

    B_shape = (cfg["N_Rec"],
               cfg["N_R"],
               tar_size,)

    if cfg["eprop_type"] == "random":  # Gaussian, variance of 1
        W["B"] = rng.normal(size=B_shape, scale=1)

    elif cfg["eprop_type"] == "adaptive":  # Gaussian, variance of 1/N
        W["B"] = rng.normal(size=B_shape, scale=np.sqrt(1 / cfg["N_R"]))

    elif cfg["eprop_type"] == "symmetric":
        W["B"] = rng.random(size=B_shape) * 2 - 1
        for r in range(cfg["N_Rec"]):
            W["B"][r] = W["out"]

    # Drop all self-looping weights. A neuron cannot be connected with itself.
    for r in range(cfg["N_Rec"]):
        np.fill_diagonal(W['W'][r, :, cfg["N_R"]:], 0)

    for wtype in ["W", "out", "bias", "B"]:
        W[wtype] = torch.tensor(W[wtype])

    return W


def update_weights(W, G, adamvars, e, cfg, it_per_e):
    adamvars['it'] += 1

    for wtype in G.keys():
        if not cfg[f"train_{wtype}"]:
            continue
        eta = cfg[f"eta_{wtype}"]
        if cfg["warmup"] and e == 0:
            eta *= adamvars['it'] / it_per_e

        eta *= (1 - cfg["eta_decay"]) ** adamvars['it']

        if cfg["Optimizer"] == "Adam":
            adamvars[f"m{wtype}"] = (cfg["adam_beta1"] * adamvars[f"m{wtype}"]
                                     + (1 - cfg["adam_beta1"]) * G[wtype])
            adamvars[f"v{wtype}"] = (cfg["adam_beta2"] * adamvars[f"v{wtype}"]
                                     + (1 - cfg["adam_beta2"]) * G[wtype] ** 2)

            m = adamvars[f"m{wtype}"] / (1 - cfg["adam_beta1"] ** adamvars['it'])
            v = adamvars[f"v{wtype}"] / (1 - cfg["adam_beta2"] ** adamvars['it'])
            dw = -eta * (m / (torch.sqrt(v) + cfg["adam_eps"]))

        elif cfg["Optimizer"] == "RAdam":

            adamvars[f"m{wtype}"] = (cfg["adam_beta1"] * adamvars[f"m{wtype}"]
                                     + (1 - cfg["adam_beta1"]) * G[wtype])

            adamvars[f"v{wtype}"] = (1 / cfg["adam_beta2"] * adamvars[f"v{wtype}"]
                                     + (1 - cfg["adam_beta2"]) * G[wtype] ** 2)

            m = adamvars[f"m{wtype}"] / (1 - cfg["adam_beta1"] ** adamvars['it'])

            rinf = 2 / (1 - cfg["adam_beta2"]) - 1

            rho = (rinf
                   - 2 * adamvars['it'] * cfg["adam_beta2"] ** adamvars['it'] / (1 - cfg["adam_beta2"] ** adamvars['it']))

            if rho > 4:

                l = torch.sqrt((1 - cfg["adam_beta2"] ** adamvars['it']) / (adamvars[f"v{wtype}"] + cfg["adam_eps"]))

                upper = (rho - 4) * (rho - 2) * rinf

                lower = (rinf - 4) * (rinf - 2) * rho

                r = np.sqrt(upper / lower)

                dw = -eta * m * r * l

        elif cfg["Optimizer"] == "Momentum":
            adamvars[f"m{wtype}"] = cfg["adam_beta1"] * adamvars[f"m{wtype}"] - eta * G[wtype]

            dw = adamvars[f"m{wtype}"]

        else:
            print("Warning: undefined optimizer")
            dw = -eta * m

        W[wtype] += dw

        if wtype == 'out' and cfg['eprop_type'] == 'adaptive':
            W['B'] += dw

            W['out'] -= cfg["weight_decay"] * W['out']
            W['B'] -= cfg["weight_decay"] * W['B']

    if cfg["eprop_type"] == 'symmetric':
        for r in range(cfg['N_Rec']):
            W['B'][r] = W['out']

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
    M['x'] = X
    M['t'] = T

    alpha = torch.tensor(cfg["alpha"])
    kappa = torch.tensor(cfg["kappa"])
    gamma = torch.tensor(cfg["gamma"])
    rho = torch.tensor(cfg['rho'])
    thr = torch.tensor(cfg["thr"])

    for t in np.arange(0, max(n_steps.cpu().numpy())):
        start = time.time()

        is_valid = torch.logical_not(torch.any(X[:, t] == -1, axis=1))

        for r in range(cfg["N_Rec"]):
            M['va'][:, r] = (M['h'][:, r, :, None] * M['vv'][:, r]
                + (rho - (M['h'][:, r, :, None]
                                 * betas[None, r, :, None]))
                * M['va'][:, r])
            Z_in = torch.cat((M['z'][:, t, r-1] if r else X[:, t],
                              M['z'][:, t-1, r]), axis=1)

            M['vv'][:, r] = alpha * M['vv'][:, r] + Z_in[:, None, :]

            M['a'][:, r] = rho * M['a'][:, r] + M['z'][:, t-1, r]

            A = thr + betas[None, r] * M['a'][:, r]

            M['v'][:, r] = ((
                  alpha * M['v'][:, r]
                  + torch.sum(W['W'][None, r] * Z_in[:, None, :], axis=-1)
                  - M['z'][:, t-1, r] * (A if cfg["v_fix"] else thr))
                  )

            M['z'][:, t, r] = torch.where(
                torch.logical_and(t - M['tz'][:, r] > cfg["dt_refr"],
                                  M['v'][:, r] >= A),
                1,
                0)

            M['h'][:, r] = (
                ((1 / (A if cfg["v_fix"] else thr)) if not cfg["v_fix_psi"] else 1)
                # (1 / cfg["thr"])
                * gamma
                * torch.clip(
                    1 - (abs((M['v'][:, r] - A) / thr)),
                              # / (A if cfg["v_fix"] else cfg["thr"]))),
                    0,
                    None))
            M['h'][:, r] = torch.where(
                t - M['tz'][:, r] > cfg["dt_refr"],
                M['h'][:, r],
                torch.zeros_like(M['h'][:, r]))

            M['tz'][:, r] = torch.where(M['z'][:, t, r] != 0, torch.ones_like(M['tz'][:, r])*t, M['tz'][:, r])
            M['zs'][:, r] += M['z'][:, t, r] * is_valid[:, None]

            ET = (
                M['h'][:, r, :, None]
                * (M['vv'][:, r] - betas[None, r, :, None] * M['va'][:, r]))

            M['etbar'][:, r] = kappa * M['etbar'][:, r] + ET


        M['zbar'] = kappa * M['zbar'] + M['z'][:, t, -1]

        M['y'] = (
            kappa * M['y']
            + torch.sum(W['out'][None] * M['z'][:, t, -1, :, None], axis=-2)
            + W['bias'])


        M['p'][:, t] = torch.exp(M['y'] - torch.amax(M['y'],
                                 axis=1)[:, None])

        M['p'][:, t] = M['p'][:, t] / torch.sum(M['p'][:, t], axis=1)[:, None]

        D = (M['p'][:, t] - T[:, t])

        loss_pred = torch.sum(
            W['B'] * D[:, None, None],  # Checked correct (for batch size 1)
            axis=-1)


        loss_reg = (
            cfg["FR_reg"]
            # * 2 * t
            # / n_steps[:, None, None]
            * (M['zs'] / (t+1) - cfg["FR_target"]))

        M['GW'] += torch.mean(loss_pred[:, :, :, None] * M['etbar'] * is_valid[:, None, None, None], axis=0)
        M['GW'] += torch.mean(loss_reg[:, :, :, None] * is_valid[:, None, None, None], axis=0)

        M['Gout'] += torch.mean(D[:, None] * M['zbar'][:, :, None] * is_valid[:, None, None], axis=0)
        M['Gbias'] += torch.mean(D * is_valid[:, None],axis=0)


    a = torch.arange(M['p'].shape[1])
    for b in range(M['p'].shape[0]):
        M['pm'][b, a, M['p'][b].argmax(axis=1)] = 1

    M['correct'] = (M['pm'] == T).all(axis=2)
    M['ce'] = -torch.sum(T * torch.log(1e-30 + M['p']), axis=2)
    dev = M['zs'] / n_steps[:, None, None] - cfg["FR_target"]
    M['reg_error'] = 0.5 * torch.sum(
        torch.sum(dev**2, axis=0))

    if cfg["L2_reg"]:
        M['GW'] += cfg["L2_reg"] * W['W']

    G = {}
    op = torch.sum if cfg["batch_op"] == 'sum' else torch.mean
    G['W'] = M['GW']
    G['out'] = M['Gout']
    G['bias'] = M['Gbias']


    # Don't update dead weights
    G['W'][W['W']==0] = 0

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
        'mW': torch.zeros(size=(cfg["N_Rec"], cfg["N_R"], cfg["N_R"]*2)),
        'vW': torch.zeros(size=(cfg["N_Rec"], cfg["N_R"], cfg["N_R"]*2)),
        'mout': torch.zeros(size=(cfg["N_R"], tar_size)),
        'vout': torch.zeros(size=(cfg["N_R"], tar_size)),
        'mbias': torch.zeros(size=(tar_size,)),
        'vbias': torch.zeros(size=(tar_size,)),
        'it': 0
    }


def initialize_betas(cfg):
    rng = np.random.default_rng(seed=cfg["seed"])
    betas = np.zeros(
        shape=(cfg["N_Rec"] * cfg["N_R"]))

    betas[:int(betas.size * cfg["fraction_ALIF"])] = cfg["beta"]
    rng.shuffle(betas)

    betas = betas.reshape((cfg["N_Rec"], cfg["N_R"],))
    betas = torch.tensor(betas)
    return betas


def conn_t_idxs(t, track_synapse):
    if track_synapse:
        return t-1, t
    return 0, 0


def initialize_W_log(cfg, W, sample_size=100):
    W_log = {}

    rng = np.random.default_rng(seed=cfg['seed'])
    W_log['Cross-entropy'] = {'train': [], 'val': []}
    W_log['Error (reg)'] = {'train': [], 'val': []}
    W_log['Mean Hz'] = {'train': [], 'val': []}
    W_log['Percentage wrong'] = {'train': [], 'val': []}

    for wtype, w in W.items():
        wf = w.flatten()
        if wtype != 'bias':
            ssize = min(wf.shape[0], sample_size)
            idxs = rng.choice(ssize, size=ssize, replace=False)[:wf.shape[0]]
        else:
            idxs = np.arange(wf.shape[0])

        W_log[f'{wtype}_idxs'] = idxs
        W_log[wtype] = np.empty(shape=(idxs.size, 0))


    return W_log


def update_W_log(W_log, Mt, Mv, W, log_id):
    # weights sample
    for wtype, w in W.items():
        wcpu = w.cpu().numpy()
        W_log[wtype] = np.append(W_log[wtype], wcpu.flatten()[W_log[f'{wtype}_idxs']][:, None], axis=1)

    for tv_type, M in (('train', Mt), ('val', Mv)):
        if M is None:  # Mv skipped

            W_log['Cross-entropy'][tv_type].append(-1)
            W_log['Mean Hz'][tv_type].append(-1)
            W_log['Percentage wrong'][tv_type].append(-1)
            W_log['Error (reg)'][tv_type].append(-1)
            continue

        X = M['x'].cpu().numpy()
        bsize = X.shape[0]
        ces = np.zeros(shape=(bsize))
        hz = np.zeros(shape=(bsize))
        pwrong = np.zeros(shape=(bsize))

        for b in range(bsize):
            arr = X[b]
            while np.any(arr[-1] == -1):
                arr = arr[:-1]
            ces[b] = np.mean(M['ce'].cpu().numpy()[b, :arr.shape[0]])
            pwrong[b] = 100-100*np.mean(M['correct'].cpu().numpy()[b, :arr.shape[0]])
            hz[b] = 1000*np.mean(M['zs'][b].cpu().numpy()/arr.shape[0])

        W_log['Cross-entropy'][tv_type].append(np.mean(ces))
        W_log['Mean Hz'][tv_type].append(np.mean(hz))
        W_log['Percentage wrong'][tv_type].append(np.mean(pwrong))
        W_log['Error (reg)'][tv_type].append(M['reg_error'])

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
        while np.any(arr[-1, 0] == -1):
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


def sample_mbatches(cfg, n_samples, tvtype):
    ret = []
    samples = np.arange(n_samples)

    # rng = np.random.default_rng(seed=cfg["seed"])
    # rng.shuffle(samples)

    while samples.shape[0]:
        ret.append(samples[:cfg[f"batch_size_{tvtype}"]])
        samples = samples[cfg[f"batch_size_{tvtype}"]:]
    return ret


def load_data(cfg):

    inps = {}
    tars = {}
    rng = np.random.default_rng(seed=cfg["seed"])
    for tvt_type in cfg['n_examples'].keys():
        inps[tvt_type] = np.load(f"{cfg['wavs_fname']}_{tvt_type}_{cfg['task']}.npy")
        tars[tvt_type] = np.load(f"{cfg['phns_fname']}_{tvt_type}_{cfg['task']}.npy")



    # mintrain = np.amin(inps['train'], axis=(0, 1))
    # maxtrain = np.ptp(inps['train'], axis=(0, 1))
    means = np.mean(inps['train'], axis=(0, 1))
    stds = np.std(inps['train'], axis=(0, 1))

    for tvt_type in cfg['n_examples'].keys():
        # Normalize [0, 1]

        inps[tvt_type] = np.where(inps[tvt_type] != -1, (inps[tvt_type] - means) / np.maximum(stds, 1e-10), -1)
        # inps[tvt_type] = np.where(inps[tvt_type] != -1, inps[tvt_type] / maxtrain, -1)


        # Add blank
        tars[tvt_type] = np.append(np.zeros_like(tars[tvt_type][:, :, :1]), tars[tvt_type], axis=2)

        shuf_idxs = np.arange(inps[tvt_type].shape[0])
        rng.shuffle(shuf_idxs)
        inps[tvt_type] = inps[tvt_type][shuf_idxs]
        tars[tvt_type] = tars[tvt_type][shuf_idxs]

        inps[tvt_type] = inps[tvt_type][:, :cfg["maxlen"]]
        tars[tvt_type] = tars[tvt_type][:, :cfg["maxlen"]]
    return inps, tars


def get_log_id():
    log_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    id_append = ""
    count = 1
    while os.path.isdir(f"../log/{log_id}{id_append}"):
        id_append = f"_{count}"
        count += 1
    return log_id + id_append


def save_checkpoint(W, cfg, log_id):
    for wtype, w in W.items():
        np.save(f"../log/{log_id}/checkpoints/{wtype}", w.cpu().numpy())


def load_checkpoint(log_id, parent_dir='log'):
    W = {}
    for subdir, _, files in os.walk(f"../{parent_dir}/{log_id}/checkpoints"):
        for filename in files:
            filepath = subdir + os.sep + filename
            W[filename[:-4]] = torch.tensor(np.load(filepath))  # cut off '.npy'
    return W
