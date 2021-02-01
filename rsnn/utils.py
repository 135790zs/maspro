
import datetime
import numpy as np
import os
import json
from scipy.interpolate import interp1d
import time
import torch
from config import cfg  # For cuda only
import csv

if cfg["cuda"]:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def initialize_model(cfg, inp_size, tar_size, batch_size, n_steps):
    M = {}

    len_syn_time = n_steps if cfg["Track_synapse"] else 1
    len_nrn_time = n_steps if cfg["Track_neuron"] else 1

    nrn_shape = (cfg["n_directions"],
                 batch_size,
                 len_nrn_time,
                 cfg["N_Rec"],
                 cfg["N_R"])
    nrn_shape_nobatch = (cfg["n_directions"],
                 len_nrn_time,
                 cfg["N_Rec"],
                 cfg["N_R"])
    nrn_shape_db = (cfg["n_directions"],
                 batch_size,
                 len_nrn_time,
                 cfg["N_Rec"],
                 cfg["N_R"]*2)
    syn_shape = (cfg["n_directions"],
                 batch_size,
                 len_syn_time,
                 cfg["N_Rec"],
                 cfg["N_R"],
                 cfg["N_R"] * 2)
    syn_shape_half = (cfg["n_directions"],
                      len_syn_time,
                      cfg["N_Rec"],
                      cfg["N_R"],
                      cfg["N_R"])

    out_shape = (batch_size,
                 len_nrn_time,
                 tar_size)

    out_shape_long = (batch_size,
                      n_steps,
                      tar_size)

    for var in ['z', 'I', 'I_in', 'I_rec', 'h', 'v', 'a', 'zbar', 'loss_pred']:
        M[var] = torch.zeros(size=nrn_shape, )
    if cfg["neuron"] == "Izhikevich":
        M['v'] = torch.ones_like(M['v']) * cfg["IzhReset"]

    for var in ['loss_reg']:
        M[var] = torch.zeros(size=nrn_shape_nobatch, )

    for var in ['vv', 'va', 'et', 'etbar']:
        M[var] = torch.zeros(size=syn_shape, )

    for var in ['GW_in', 'GW_rec']:
        M[var] = torch.zeros(size=syn_shape_half, )

    nrn_timeless = (cfg["n_directions"],
                    batch_size,
                    cfg["N_Rec"],
                    cfg["N_R"])
    M['tz'] = torch.ones(size=nrn_timeless, dtype=torch.int) * -cfg["dt_refr"]
    M['zs'] = torch.zeros(size=nrn_timeless)
    M['z_in'] = torch.zeros(size=nrn_shape_db)

    for var in ['y','d']:
        M[var] = torch.zeros(size=out_shape, )

    for var in ['p','pm']:
        M[var] = torch.zeros(size=out_shape_long, )

    M['ysub'] = torch.zeros(size=(cfg["n_directions"],
                                batch_size,
                                len_nrn_time,
                                tar_size), )

    M["ce"] = torch.zeros(size=(batch_size, n_steps,))
    M["correct"] = torch.zeros(size=(batch_size, n_steps,))

    M["Gout"] = torch.zeros(size=(cfg["n_directions"],
                                  len_syn_time,
                                  cfg["N_R"],
                                  tar_size))
    M["Gbias"] = torch.zeros(size=(len_syn_time,
                                   tar_size))

    return M


def initialize_gradients(cfg, tar_size, batch_size):
    G = {}
    G["W_in"] = torch.zeros(size=(cfg["n_directions"],
                                  batch_size,
                                  cfg["N_Rec"],
                                  cfg["N_R"],
                                  cfg["N_R"],))
    G["W_rec"] = torch.zeros(size=(cfg["n_directions"],
                                   batch_size,
                                   cfg["N_Rec"],
                                   cfg["N_R"],
                                   cfg["N_R"],))
    G["out"] = torch.zeros(size=(cfg["n_directions"],
                               batch_size,
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

    W["W_in"] = rng.normal(size=(cfg["n_directions"],
                                 cfg["N_Rec"],
                                 cfg["N_R"],
                                 cfg["N_R"],))
    # Bellec: rd.randn(n_in, n_rec) / np.sqrt(n_in)
    W['W_in'][:, 0, :, inp_size:] = 0  # From padded channels
    W['W_in'][:, 0] /= np.sqrt(inp_size)
    for r in range(cfg["N_Rec"]):
        if r > 0:
            W['W_in'][:, r] /= np.sqrt(cfg["N_R"])

    W["W_rec"] = rng.normal(size=(cfg["n_directions"],
                                  cfg["N_Rec"],
                                  cfg["N_R"],
                                  cfg["N_R"],))
    # Bellec: rd.randn(n_rec, n_rec) / np.sqrt(n_rec)
    W['W_rec'] /= np.sqrt(cfg["N_R"])


    W["out"] = rng.normal(size=(cfg["n_directions"],
                                cfg["N_R"],
                                tar_size)) / np.sqrt(tar_size)
    if cfg["uniform_dist"]:
        W["W_in"] = rng.random(size=(cfg["n_directions"],
                                 cfg["N_Rec"],
                                 cfg["N_R"],
                                 cfg["N_R"],)) * 2 - 1

        W['W_in'][:, 0, :, inp_size:] = 0  # From padded channels
        W['W_in'][:, 0] /= np.sqrt(inp_size)

        for r in range(cfg["N_Rec"]):
            if r > 0:
                W['W_in'][:, r] /= np.sqrt(cfg["N_R"])

        W["W_rec"] = rng.random(size=(cfg["n_directions"],
                                      cfg["N_Rec"],
                                      cfg["N_R"],
                                      cfg["N_R"],)) * 2 - 1
        W['W_rec'] /= np.sqrt(cfg["N_R"])


        W["out"] = rng.random(size=(cfg["n_directions"],
                                    cfg["N_R"],
                                    tar_size)) * 2 - 1 / np.sqrt(tar_size)
        W['out'] /= np.sqrt(tar_size)

    for wtype in W.keys():
        W[wtype] *= cfg["weightscale"]

    if cfg["one_to_one_output"]:
        W["out"] *= 0
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W["out"][s, :, :], 1.)



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
                W["B"][s, r] = W["out"][s]

    # Drop all self-looping weights. A neuron cannot be connected with itself.
    for r in range(cfg["N_Rec"]):
        for s in range(cfg["n_directions"]):
            np.fill_diagonal(W['W_rec'][s, r], 0)

    W['W_rec'] = np.where(rng.random(W['W_rec'].shape) < cfg["dropout"],
                         0,
                         W['W_rec'])

    for wtype in ["W_in", "W_rec", "out", "bias", "B"]:
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
        for s in range(cfg['n_directions']):
            for r in range(cfg['N_Rec']):
                W['B'][s, r] = W['out'][s]

    return W, adamvars


def eprop(cfg, X, T, betas, W, grad=True):

    start = time.time()

    X, T = interpolate_inputs(cfg=cfg,
                                 inp=X,
                                 tar=T,
                                 stretch=cfg["Repeats"])

    X, T = trim_samples(X=X, T=T)

    n_steps = count_lengths(X=X)

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

    M['x'] = X[None, :]  # Insert subnetwork dimension
    M['t'] = T

    if cfg["n_directions"] == 2:
        M['x'] = torch.cat((M['x'], M['x']))
        for b in range(M['x'].shape[1]):
            M['x'][1, b, :n_steps[b]] = torch.fliplr(M['x'][0, b, :n_steps[b]])

    for t in np.arange(0, max(n_steps.cpu().numpy())):
        prev_syn_t, curr_syn_t = conn_t_idxs(track_synapse=cfg['Track_synapse'], t=t)
        prev_nrn_t, curr_nrn_t = conn_t_idxs(track_synapse=cfg['Track_neuron'], t=t)
        is_valid = torch.logical_not(torch.any(M['x'][0, :, t] == -1, axis=1))

        for r in range(cfg["N_Rec"]):  # TODO: Can overwrite r instead of appending, except Z
            # TODO: See which is_valid masks can be removed


            if grad and cfg["neuron"] in ["ALIF", "STDP-ALIF"]:
                M['va'][:, :, curr_syn_t, r] = (
                    M['h'][:, :, prev_nrn_t, r, :, None] * M['vv'][:, :, prev_syn_t, r]
                    + (cfg["rho"] - (M['h'][:, :, prev_nrn_t, r, :, None]
                                     * betas[:, None, r, :, None]))
                    * M['va'][:, :, prev_syn_t, r])
            elif grad and cfg["neuron"] == "Izhikevich":
                oldva = M['va'][:, :, prev_syn_t, r]
                M['va'][:, :, curr_syn_t, r] = (
                    cfg["IzhA1"] * (1 - M['z'][:, :, prev_nrn_t, r, :, None]) * M['vv'][:, :, prev_syn_t, r]
                    + (1 + cfg["IzhA2"]) * oldva)

            Z_prev_layer = (M['z'][:, :, curr_nrn_t, r-1] if r
                            else M['x'][:, :, t])
            Z_prev_time = M['z'][:, :, prev_nrn_t, r]

            # Z_in is incoming at current t, so from last t.
            M['z_in'][:, :, curr_nrn_t, r] = torch.cat((Z_prev_layer, Z_prev_time), axis=2)

            if grad and cfg["neuron"] == "ALIF":
                M['vv'][:, :, curr_syn_t, r] = (
                    cfg["alpha"]
                    * M['vv'][:, :, prev_syn_t, r]
                    + M['z_in'][:, :, curr_nrn_t, r, None, :]
                    )
            elif grad and cfg["neuron"] == "STDP-ALIF":
                M['vv'][:, :, curr_syn_t, r] = (
                    cfg["alpha"]
                    * M['vv'][:, :, prev_syn_t, r]
                    * (1 - M['z'][:, :, prev_nrn_t, r] - torch.where(
                            M['tz'][:, :, r] == t-1 - cfg["dt_refr"],
                            1,
                            0))[..., None]
                    + M['z_in'][:, :, curr_nrn_t, r, None, :]
                    )
            elif grad and cfg["neuron"] == "Izhikevich":
                M['vv'][:, :, curr_syn_t, r] = (
                    (1 - M['z'][:, :, prev_nrn_t, r, :, None])
                    * (1 + (2 * cfg["IzhV1"] * M['v'][:, :, prev_nrn_t, r, :, None] + cfg["IzhV2"]))
                    * M['vv'][:, :, prev_syn_t, r]
                    - oldva
                    + M['z_in'][:, :, curr_nrn_t, r, None, :])

            # Can contract this further!
            M['I_in'][:, :, curr_nrn_t, r] = torch.sum(W['W_in'][:, None, r] * Z_prev_layer[:, :, None, :], axis=-1) # Checked correct

            M['I_rec'][:, :, curr_nrn_t, r] = torch.sum(W['W_rec'][:, None, r] * Z_prev_time[:, :, None, :], axis=-1) # Checked correct

            M['I'][:, :, curr_nrn_t, r] = M['I_in'][:, :, curr_nrn_t, r] + M['I_rec'][:, :, curr_nrn_t, r]

            if cfg["neuron"] in ["ALIF", "STDP-ALIF"]:
                M['a'][:, :, curr_nrn_t, r] = (cfg["rho"] * M['a'][:, :, prev_nrn_t, r]
                                      + M['z'][:, :, prev_nrn_t, r])

            elif cfg["neuron"] == "Izhikevich":
                at = M['a'][:, :, prev_nrn_t, r] + 2 * M['z'][:, :, prev_nrn_t, r]
                M['a'][:, :, curr_nrn_t, r] = (
                    at
                    + cfg["IzhA1"] * (M['v'][:, :, prev_nrn_t, r] - (M['v'][:, :, prev_nrn_t, r] + cfg["thr"]) * M['z'][:, :, prev_nrn_t, r])
                    + cfg["IzhA2"] * at)

            A = cfg["thr"] + betas[:, None, r] * M['a'][:, :, curr_nrn_t, r]

            if cfg["neuron"] == "ALIF":
                M['v'][:, :, curr_nrn_t, r] = ((
                      cfg["alpha"] * M['v'][:, :, prev_nrn_t, r]
                      + M['I'][:, :, curr_nrn_t, r]
                      - M['z'][:, :, prev_nrn_t, r] * (A if cfg["v_fix"] else cfg["thr"]))
                      )
            elif cfg["neuron"] == "STDP-ALIF":
                M['v'][:, :, curr_nrn_t, r] = ((
                      cfg["alpha"] * M['v'][:, :, prev_nrn_t, r]
                      + M['I'][:, :, curr_nrn_t, r]
                      - M['z'][:, :, prev_nrn_t, r] * cfg["alpha"] * M['v'][:, :, prev_nrn_t, r]
                      - cfg["alpha"] * M['v'][:, :, prev_nrn_t, r] * torch.where(
                            M['tz'][:, :, r] == t - cfg["dt_refr"],
                            1,
                            0))
                      )
            elif cfg["neuron"] == "Izhikevich":
                vt = M['v'][:, :, prev_nrn_t, r] - (M['v'][:, :, prev_nrn_t, r] - cfg["IzhReset"]) * M['z'][:, :, prev_nrn_t, r]
                M['v'][:, :, curr_nrn_t, r] = (
                      vt
                      + cfg["IzhV1"]*vt**2
                      + cfg["IzhV2"]*vt
                      + cfg["IzhV3"]
                      - (M['a'][:, :, prev_nrn_t, r] + 2 * M['z'][:, :, prev_nrn_t, r])
                      + M['I'][:, :, curr_nrn_t, r]
                      )

            if cfg["neuron"] in ["ALIF", "STDP-ALIF"]:
                M['z'][:, :, curr_nrn_t, r] = torch.where(
                    torch.logical_and(t - M['tz'][:, :, r] > cfg["dt_refr"],
                                      M['v'][:, :, curr_nrn_t, r] >= A),
                    1,
                    0)
            elif cfg["neuron"] == "Izhikevich":
                M['z'][:, :, curr_nrn_t, r] = torch.where(
                    M['v'][:, :, curr_nrn_t, r] >= cfg["thr"],
                    1,
                    0)
                M['v'][:, :, curr_nrn_t, r] = torch.where(
                    M['z'][:, :, curr_nrn_t, r] == 1,
                    torch.ones_like(M['v'][:, :, curr_nrn_t, r]) * cfg["IzhReset"],
                    M['v'][:, :, curr_nrn_t, r])
                M['a'][:, :, curr_nrn_t, r] = torch.where(
                    M['z'][:, :, curr_nrn_t, r] == 1,
                    M['a'][:, :, curr_nrn_t, r] + 2,
                    M['a'][:, :, curr_nrn_t, r])

            M['tz'][:, :, r] = torch.where(M['z'][:, :, curr_nrn_t, r] != 0,
                                           torch.ones_like(M['tz'][:, :, r])*t,
                                           M['tz'][:, :, r])
            M['zs'][:, :, r] += M['z'][:, :, curr_nrn_t, r] * is_valid[None, :, None]
            if not grad:
                continue

            if cfg["neuron"] in ['ALIF', "STDP-ALIF"]:
                M['h'][:, :, curr_nrn_t, r] = (
                    ((1 / (A if cfg["v_fix"] else cfg["thr"])) if not cfg["v_fix_psi"] else 1)
                    # (1 / cfg["thr"])
                    * cfg["gamma"]
                    * torch.clip(
                        1 - (abs((M['v'][:, :, curr_nrn_t, r] - A) / cfg["thr"])),
                        0,
                        None))
            elif cfg["neuron"] == 'Izhikevich':
                cfg["gamma"] * torch.exp(
                    (torch.clip(M['v'][:, :, curr_nrn_t, r], None, cfg["IzhPsi"]) - cfg["IzhPsi"])
                    / cfg["IzhPsi"])

            if cfg["neuron"] == "ALIF":
                M['h'][:, :, curr_nrn_t, r] = torch.where(
                    t - M['tz'][:, :, r] >= cfg["dt_refr"],
                    M['h'][:, :, curr_nrn_t, r],
                    torch.zeros_like(M['h'][:, :, curr_nrn_t, r]))
            elif cfg["neuron"] == "STDP-ALIF":
                M['h'][:, :, curr_nrn_t, r] = torch.where(
                    t - M['tz'][:, :, r] >= cfg["dt_refr"],
                    M['h'][:, :, curr_nrn_t, r],
                    torch.ones_like(M['h'][:, :, curr_nrn_t, r]) * cfg["gamma"])



            if cfg["neuron"] in ['ALIF', "STDP-ALIF"]:
                M['et'][:, :, curr_syn_t, r] = (
                    M['h'][:, :, curr_nrn_t, r, :, None]
                    * (M['vv'][:, :, curr_syn_t, r]
                       - betas[:, None, r, :, None] * M['va'][:, :, curr_syn_t, r])
                    )
            elif cfg["neuron"] == 'Izhikevich':
                M['et'][:, :, curr_syn_t, r] = (
                    M['h'][:, :, curr_nrn_t, r, :, None] * M['vv'][:, :, curr_syn_t, r])

            M['etbar'][:, :, curr_syn_t, r] = (
                cfg["kappa"]
                * M['etbar'][:, :, prev_syn_t, r] + M['et'][:, :, curr_syn_t, r]
                )
            M['zbar'][:, :, curr_nrn_t, r] = (
                cfg["kappa"]
                * M['zbar'][:, :, prev_nrn_t, r] + M['z'][:, :, curr_nrn_t, r]
                )

        M['ysub'][:, :, curr_nrn_t] = (
            torch.sum(
                W['out'][:, None] * M['z'][:, :, curr_nrn_t, -1, :, None],
                axis=-2)
            + cfg["kappa"] * M['ysub'][:, :, prev_nrn_t])

        M['y'][:, curr_nrn_t] = torch.sum(M['ysub'][:, :, curr_nrn_t], axis=0) + W['bias']

        M['p'][:, t] = torch.exp(M['y'][:, curr_nrn_t]
                                    - torch.amax(M['y'][:, curr_nrn_t],
                                 axis=1)[:, None])

        M['p'][:, t] = M['p'][:, t] / torch.sum(M['p'][:, t], axis=1)[:, None]

        if not grad:  # Next steps all have to do with gradient calculation
            continue

        M['d'][:, curr_nrn_t] = (M['p'][:, t] - T[:, t])

        M['loss_pred'][:, :, curr_nrn_t] = torch.sum(
            W['B'][:, None, :, :] * M['d'][None, :, curr_nrn_t, None, None, :],  # Checked correct (for batch size 1)
            axis=-1)

        M['GW_in'][:, curr_syn_t] += torch.mean(
            M['loss_pred'][:, :, curr_nrn_t, :, :, None]
            * M['etbar'][:, :, curr_syn_t, :, :, :cfg["N_R"]], axis=1)
        M['GW_rec'][:, curr_syn_t] += torch.mean(
            M['loss_pred'][:, :, curr_nrn_t, :, :, None]
            * M['etbar'][:, :, curr_syn_t, :, :, cfg["N_R"]:], axis=1)


        M['loss_reg'][:, curr_nrn_t] = torch.mean(
            cfg["FR_reg"]
            # * 2 * t  # Initial means are less informative
            # / n_steps[None, :, None, None] ** 2  # Square corrects previous term
            * (M['zs'] / (t+1) - cfg["FR_target"]) * is_valid[None, :, None, None], axis=1)

        M['GW_in'][:, curr_syn_t] += M['loss_reg'][:, curr_nrn_t, :, :, None]
        M['GW_rec'][:, curr_syn_t] += M['loss_reg'][:, curr_nrn_t, :, :, None]

        if cfg["L2_reg"]:
            M['GW_in'][:, curr_syn_t] += cfg["L2_reg"] * W['W_in'] * is_valid[:, None, None]
            M['GW_rec'][:, curr_syn_t] += cfg["L2_reg"] * W['W_rec'] * is_valid[:, None, None]

        M['Gout'][:, curr_syn_t] += torch.mean(
            M['d'][None, :, curr_nrn_t, None]
            * M['zbar'][:, :, curr_nrn_t, -1, :, None]
            * is_valid[None, :, None, None], axis=1)
        M['Gbias'][curr_syn_t] += torch.mean(torch.sum(M['d'][:, curr_nrn_t] * is_valid[None, :, None],
                                                       axis=0), axis=0)

    a = torch.arange(M['p'].shape[1])
    for b in range(M['p'].shape[0]):
        M['pm'][b, a, M['p'][b].argmax(axis=1)] = 1

    M['correct'] = (M['pm'] == M['t']).all(axis=2)

    M['ce'] = -torch.sum(M['t'] * torch.log(1e-30 + M['p']), axis=2)
    M['reg_error'] = 0.5 * torch.sum(
        torch.mean(
            (M['zs'] / n_steps[None, :, None, None] - cfg["FR_target"]),
            axis=1)) ** 2

    if grad:
        G = {}
        # Sum over time
        G['W_in'] = torch.sum(M['GW_in'], axis=1)
        G['W_rec'] = torch.sum(M['GW_rec'], axis=1)
        G['out'] = torch.sum(M['Gout'], axis=1)
        G['bias'] = torch.sum(M['Gbias'], axis=0)

        # Don't update dead weights
        G['W_in'][W['W_in']==0] = 0
        G['W_rec'][W['W_rec']==0] = 0

        return G, M, n_steps.cpu().numpy()

    return None, M, n_steps.cpu().numpy()


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
        'mW_in': torch.zeros(size=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"])),
        'vW_in': torch.zeros(size=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"])),
        'mW_rec': torch.zeros(size=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"])),
        'vW_rec': torch.zeros(size=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"])),
        'mout': torch.zeros(size=(cfg["n_directions"], cfg["N_R"], tar_size)),
        'vout': torch.zeros(size=(cfg["n_directions"], cfg["N_R"], tar_size)),
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


def get_avgs(M, n_steps):
    A = {}
    batchsize = n_steps.shape[0]

    ces = np.zeros(shape=(batchsize,))
    hzs = np.zeros(shape=(batchsize,))
    regerrs = np.zeros(shape=(batchsize,))
    accs = np.zeros(shape=(batchsize,))

    for b in range(batchsize):
        ces[b] = np.mean(M['ce'][b, :n_steps[b]].cpu().numpy())
        hzs[b] = np.mean(M['zs'][:, b].cpu().numpy() / n_steps[b])
        regerrs[b] = M['reg_error'].cpu().numpy()
        accs[b] = np.mean(M['correct'][b, :n_steps[b]].cpu().numpy())

    A['ce'] = np.mean(ces)
    A['hz'] = np.mean(hzs)
    A['regerr'] = np.mean(regerrs)
    A['acc'] = np.mean(accs)

    return A


def update_W_log(W_log, W, log_id, avgs_t, avgs_v):
    # weights sample
    for wtype, w in W.items():
        wcpu = w.cpu().numpy()
        W_log[wtype] = np.append(W_log[wtype], wcpu.flatten()[W_log[f'{wtype}_idxs']][:, None], axis=1)

    if not os.path.exists(f'../log/{log_id}/tracker/'):
        os.mkdir(f'../log/{log_id}/tracker/')

    for key in ['Error (reg)', 'Percentage wrong', 'Mean Hz', 'Cross-entropy']:

        if not os.path.exists(f'../log/{log_id}/tracker/{key}.csv'):
            with open(f'../log/{log_id}/tracker/{key}.csv', 'w', newline='') as new_csv:
                varname_writer = csv.writer(new_csv,
                                        delimiter=',',
                                        quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                varname_writer.writerow(['train', 'val'])

    for tv_type, avgs in (('train', avgs_t), ('val', avgs_v)):
        if avgs is None:  # Mv skipped

            W_log['Cross-entropy'][tv_type].append(-1)
            W_log['Mean Hz'][tv_type].append(-1)
            W_log['Percentage wrong'][tv_type].append(-1)
            W_log['Error (reg)'][tv_type].append(-1)
            continue

        W_log['Cross-entropy'][tv_type].append(avgs['ce'])
        W_log['Mean Hz'][tv_type].append(avgs['hz'])
        W_log['Percentage wrong'][tv_type].append(100 - 100 * avgs['acc'])
        W_log['Error (reg)'][tv_type].append(avgs['regerr'])

    for key in ['Error (reg)', 'Percentage wrong', 'Mean Hz', 'Cross-entropy']:
        with open(f'../log/{log_id}/tracker/{key}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile,
                                   delimiter=',',
                                   quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([W_log[key]['train'][-1], W_log[key]['val'][-1]])
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

    rng = np.random.default_rng(seed=cfg["seed"])
    rng.shuffle(samples)

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

        shuf_idxs = np.arange(inps[tvt_type].shape[0])
        rng.shuffle(shuf_idxs)
        inps[tvt_type] = inps[tvt_type][shuf_idxs]
        tars[tvt_type] = tars[tvt_type][shuf_idxs]

        inps[tvt_type] = inps[tvt_type][:, :cfg["maxlen"]]
        tars[tvt_type] = tars[tvt_type][:, :cfg["maxlen"]]

    if cfg["test_on_val"]:
        inps['test'] = inps['val']
        tars['test'] = tars['val']

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


