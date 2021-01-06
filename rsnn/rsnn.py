import time
import numpy as np
from config import cfg as CFG
import utils as ut
import vis


def network(cfg, inp, tar, W_rec, W_out, b_out, B, adamvars, eta):

    inp = np.pad(array=inp, mode='edge', pad_width=((0, cfg["delay"]), (0, 0)))

    n_steps = inp.shape[0]
    M = ut.initialize_model(cfg=cfg, length=n_steps, tar_size=tar.shape[-1])

    M["T"] = tar
    M["T"] = np.pad(array=M["T"], mode='edge', pad_width=((cfg["delay"], 0), (0, 0)))

    # X0 is the regular input sequence (T ✕ N_I)
    M[f"X0"] = inp

    # In a bidirectional network, X1 is the time-reversed sequence for
    # the second network.
    if cfg["n_directions"] == 2:
        M[f"X1"] = np.flip(inp, axis=0)

    for t in range(n_steps-1):
        start = time.time()
        for s in range(cfg["n_directions"]):

            # Input is nonzero for first layer
            for r in range(cfg['N_Rec']):
                M = ut.process_layer(cfg=cfg,
                                     M=M,
                                     t=t,
                                     s=s,
                                     r=r,
                                     W_rec=W_rec[s, r])

            M['Y'][s, t] = (
                cfg["kappa"] * (M['Y'][s, t-1] if t > 0 else 0)
                + np.sum(W_out[s] * M['Z'][s, t, -1], axis=1)
                + b_out[s])

        # Shared output layer, so step out of subnetwork loop.
        Ysum = np.sum(M['Y'][:, t], axis=0)

        M['P'][t] = ut.eprop_P(cfg=cfg, Y=Ysum)

        M['Pmax'][t, M['P'][t].argmax()] = 1

        M['CE'][t] = -np.sum(M['T'][t] * np.log(1e-30 + M['P'][t]))

        if cfg["Track_synapse"]:
            curr_t = t
        else:
            curr_t = 0

        M['D'][t] = M['P'][t] - M['T'][t]

        # Calculate weight updates for all subnetworks
        for s in range(cfg["n_directions"]):

            M['L_std'][s, t] = np.einsum("rjk, k -> rj",
                                         B[s],
                                         M['D'][t])  # Checked correct

            if t:
                M['spikerate'][s, t] = np.mean(M['Z'][s, :t], axis=0)

                M['L_reg'][s] = (cfg["FR_reg"]
                                 * (M['spikerate'][s] - cfg["FR_target"]))

            else:
                M['L_reg'][s, t] = 0

            for r in range(cfg["N_Rec"]):
                M[f'gW'][s, curr_t, r] += ut.einsum(a=M['L_std'][s, t, r],
                                                    b=M['ETbar'][s, curr_t, r])  # Checked correct

            M[f'gW'][s, curr_t] += np.repeat(
                M['L_reg'][s, t][:, :, np.newaxis],
                repeats=cfg["N_R"]*2,
                axis=2)  # Checked correct

            if cfg["update_W_out"]:
                M[f'gW_out'][s, curr_t] += np.outer(M['D'][t], M['Zbar'][s, t, -1])

            if cfg["update_bias"]:
                M[f'gb_out'][s, curr_t] += M['D'][t]
        if cfg["plot_graph"]:
            vis.plot_graph(
                cfg=cfg, M=M, s=s, t=t, W_rec=W_rec, W_out=W_out)
            time.sleep(0.5)

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out, eta, B, tvt_type, adamvars, e, log_id, start_time=None):
    batch_err = 0
    batch_perc_wrong = 0
    batch_spikerate = 0

    batch_gW = {
        'W': np.zeros(
            shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'W_out': np.zeros(
            shape=(cfg["n_directions"], tars.shape[-1], cfg["N_R"],)),
        'b_out': np.zeros(
            shape=(cfg["n_directions"], tars.shape[-1],)),
        'B': np.zeros(
            shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], tars.shape[-1],)),
    }
    for b in range(inps.shape[0]):
        if not cfg["verbose"]:
            print(f"ep {e}/{cfg['Epochs']}; \t "
                  f"{'  ' if tvt_type == 'val' else ''}"
                  f"{tvt_type} {b}/{inps.shape[0]}  ", end='\r')
        if cfg["verbose"]:
            plustime, remtime = ut.get_elapsed_time(cfg=cfg, b=b, start_time=start_time, batch_size=inps.shape[0], e=e)

            print((f"({log_id})\t+{plustime}{remtime}\t"
                   f"Epoch {e}/{cfg['Epochs']-1}\t" if tvt_type != 'test'
                   else '\t'),
                  f"{'  ' if tvt_type == 'val' else ''}{tvt_type} "
                  f"sample {b+1}/{inps.shape[0]}\t",
                  end='\r' if b < inps.shape[0]-1 else '\n')

        this_tars = tars[b]

        # Crop silence off of data
        while this_tars[-1, 0] == 1:
            this_tars = this_tars[:-1]

            if this_tars.size == 0:
                # Process failed: was 0 everywhere. Continue with original
                this_tars = tars[b]
                break

        this_inps = inps[b, :this_tars.shape[0]]

        this_inps, this_tars = ut.interpolate_inputs(inp=this_inps,
                                                     tar=this_tars,
                                                     stretch=cfg["Repeats"])

        final_model = network(
            cfg=cfg,
            inp=this_inps,
            tar=this_tars,
            W_rec=W_rec,
            W_out=W_out,
            b_out=b_out,
            B=B,
            adamvars=adamvars,
            eta=eta)

        if (cfg['plot_state'] and b == 0 and tvt_type == "train"
            and cfg["plot_state_interval"] and e % cfg["plot_state_interval"] == 0):
            vis.plot_state(cfg=cfg,
                           M=final_model,
                           B=B,
                           W_rec=W_rec,
                           W_out=W_out,
                           b_out=b_out,
                           e=e,
                           log_id=log_id)
        if (cfg['plot_state'] and b == 0 and tvt_type == "train"
            and cfg["plot_pair_interval"] and e % cfg["plot_pair_interval"] == 0):
            vis.plot_pair(cfg=cfg,
                           M=final_model,
                           B=B,
                           W_rec=W_rec,
                           W_out=W_out,
                           b_out=b_out,
                           e=e,
                           log_id=log_id)

        # Aggregate the mean batch error to the aggregator.
        # Dividing by batch size to get batch mean.
        batch_spikerate += np.mean(final_model["spikerate"]) / inps.shape[0]
        batch_err += np.mean(final_model["CE"]) / inps.shape[0]
        batch_perc_wrong += np.mean(
            np.max(np.abs(final_model["Pmax"] - final_model["T"]),
                   axis=1)) / inps.shape[0]



        for w_type in ['W', 'W_out', 'b_out']:
            gw = np.sum(final_model[f'g{w_type}'], axis=1)  # Checked correct
            batch_gW[w_type] += gw# / inps.shape[0]  # Divide to correct for batch size?

    return batch_err, batch_perc_wrong, batch_gW, batch_spikerate


def main(cfg):

    start_time = time.time()

    # Load data
    inps = {}
    tars = {}

    for tvt_type in cfg['n_examples'].keys():
        inps[tvt_type] = np.load(f"{cfg['wavs_fname']}_{tvt_type}_{cfg['task']}.npy")
        # Normalize [0, 1]

        # inps[tvt_type] = ((inps[tvt_type] - np.min(inps[tvt_type]))
        #                   / np.ptp(inps[tvt_type]))
        # print(np.min(inps[tvt_type]))
        # print(np.max(inps[tvt_type]))
        inps[tvt_type] -= np.min(inps[tvt_type], axis=1)[:, np.newaxis, :]
        inps[tvt_type] /= np.max(inps[tvt_type], axis=1)[:, np.newaxis, :]

        tars[tvt_type] = np.load(f"{cfg['phns_fname']}_{tvt_type}_{cfg['task']}.npy")


    print("N_train:", inps['train'].shape[0], "N_val:", inps['val'].shape[0])
    log_id = ut.get_log_id()

    rng = np.random.default_rng(seed=cfg["seed"])

    ut.prepare_log(cfg=cfg, log_id=log_id)

    terrs = np.zeros(shape=(cfg["Epochs"]))
    verrs = np.ones(shape=(cfg["Epochs"])) * -1
    verr = None
    percs_wrong_t = np.zeros(shape=(cfg["Epochs"]))
    percs_wrong_v = np.ones(shape=(cfg["Epochs"])) * -1
    etas = np.ones(shape=(cfg["Epochs"])) * -1
    spikerates = np.ones(shape=(cfg["Epochs"])) * -1

    optVerr = None
    W = ut.initialize_weights(cfg=cfg,
                              inp_size=inps['train'].shape[-1],
                              tar_size=tars['train'].shape[-1])

    DW = ut.initialize_DWs(cfg=cfg,
                           inp_size=inps['train'].shape[-1],
                           tar_size=tars['train'].shape[-1])

    adamvars = ut.init_adam(cfg=cfg, tar_size=tars['train'].shape[-1])

    # Print size of system in memory
    if cfg["verbose"]:
        M = ut.initialize_model(cfg=cfg,
                                length=cfg["maxlen"]*cfg["Repeats"],
                                tar_size=tars['train'].shape[-1])
        print("\nTOTAL NETWORK SIZE:",
              sum([sum([v.size for k, v in D.items()])
                   for D in [M, W, DW, adamvars]]))
        del M


    for e in range(0, cfg["Epochs"]):
        if cfg["Track_weights"]:
            ep_curr = e
            ep_incr = 1
        else:
            ep_curr = 0
            ep_incr = 0

        if verr is None or cfg["eta_init_loss"] <= 0:
            eta = cfg["eta_init"]
        else:
            eta = min(cfg["eta_init"],
                      cfg["eta_init"] * (verr / cfg["eta_init_loss"]) ** cfg["eta_slope"])
        if e+1 < cfg["ramping"]:
            eta *= (e+1)/cfg["ramping"]

        etas[e] = eta

        # Make batch
        randidxs = rng.integers(inps['train'].shape[0],
                                size=cfg["batch_size_train"])

        terr, perc_wrong_t, gW, spikerate = feed_batch(
            tvt_type='train',
            cfg=cfg,
            inps=inps['train'][randidxs],
            tars=tars['train'][randidxs],
            W_rec=W['W'][ep_curr],
            W_out=W['W_out'][ep_curr],
            b_out=W['b_out'][ep_curr],
            B=W['B'][ep_curr],
            adamvars=adamvars,
            e=e,
            log_id=log_id,
            eta=eta,
            start_time=start_time)

        terrs[e] = terr
        percs_wrong_t[e] = perc_wrong_t
        spikerates[e] = spikerate

        # Validation
        if e % cfg["val_every_E"] == 0:
            randidxs = rng.integers(inps['val'].shape[0],
                                    size=cfg["batch_size_val"])
            verr, perc_wrong_v, _, _ = feed_batch(
                tvt_type='val',
                cfg=cfg,
                inps=inps['val'][randidxs],
                tars=tars['val'][randidxs],
                W_rec=W['W'][ep_curr],
                W_out=W['W_out'][ep_curr],
                b_out=W['b_out'][ep_curr],
                B=W['B'][ep_curr],
                adamvars=adamvars,
                e=e,
                log_id=log_id,
                eta=eta,
                start_time=start_time)
            verrs[e] = verr
            percs_wrong_v[e] = perc_wrong_v


            # Save best weights
            if optVerr is None or verr < optVerr:
                if cfg["verbose"]:
                    print(f"\nLowest val error ({verr:.3f}) found at epoch {e}!\n")
                optVerr = verr
                ut.save_weights(W=W, epoch=e if cfg["Track_weights"] else 0,
                                log_id=log_id)

            # Interpolate missing verrs
            verrs[:e+1] = ut.interpolate_verrs(verrs[:e+1])
            percs_wrong_v[:e+1] = ut.interpolate_verrs(percs_wrong_v[:e+1])

        if cfg["plot_run_interval"] and e % cfg["plot_run_interval"] == 0:
            vis.plot_run(cfg=cfg, terrs=terrs, percs_wrong_t=percs_wrong_t,
                         verrs=verrs, percs_wrong_v=percs_wrong_v,
                         etas=etas, spikerates=spikerates,
                         W=W, epoch=e, log_id=log_id, inp_size=inps['train'].shape[-1])
        # Dont update weights in last epoch
        if e == cfg['Epochs'] - 1:
            break


        L2_reg = cfg["L2_reg"] * np.linalg.norm(W['W'][ep_curr].flatten())

        # Calculate DWs
        for wtype in W.keys():
            gW[wtype] += L2_reg
            if wtype == "B":  # There's no DW for B. B is updated differently.
                continue
            if cfg["optimizer"] == "Adam":
                # Update Adam for W, W_out, b_out
                adamvars[f'm{wtype}'] = (
                    cfg["adam_beta1"] * adamvars[f'm{wtype}']
                    + (1 - cfg["adam_beta1"]) * gW[wtype])
                adamvars[f'v{wtype}'] = (
                    cfg["adam_beta2"] * adamvars[f'v{wtype}']
                    + (1 - cfg["adam_beta2"]) * gW[wtype] ** 2)
                m = adamvars[f'm{wtype}'] / (1 - cfg["adam_beta1"])
                v = adamvars[f'v{wtype}'] / (1 - cfg["adam_beta2"])

                # Calculate DWs
                DW[wtype] = (-(eta if wtype != "b_out" or cfg["eta_b_out"] is None else cfg["eta_b_out"])
                             * (m / (np.sqrt(v) + cfg["adam_eps"])))
            elif cfg["optimizer"] == "SGD":
                DW[wtype] = (-(eta if wtype != "b_out" or cfg["eta_b_out"] is None else cfg["eta_b_out"])
                             * gW[wtype])

        if not cfg["update_input_weights"]:
            DW['W'][:, 0, :, :inps['train'].shape[-1]] = 0

        if not cfg["update_dead_weights"]:
            DW['W'][W["W"][ep_curr] == 0] = 0

        # Apply DWs
        for wtype in W.keys():

            # Update weights
            if not cfg["update_bias"] and wtype == "b_out":
                W[wtype][ep_curr+ep_incr] = W[wtype][ep_curr]
                continue
            if not cfg["update_W_out"] and wtype == "W_out":
                W[wtype][ep_curr+ep_incr] = W[wtype][ep_curr]
                continue
            if not cfg["update_W"] and wtype == "W":
                W[wtype][ep_curr+ep_incr] = W[wtype][ep_curr]
                continue

            if wtype == "B":

                # "Global" is already next by definition.

                if cfg["eprop_type"] in ["global", "random"]:
                    W["B"][ep_curr+ep_incr, :] = W["B"][ep_curr, :]

                elif cfg["eprop_type"] == "symmetric":
                    for s in range(cfg["n_directions"]):  # TODO: Get rid of s?
                        W["B"][ep_curr+ep_incr, s] = (W["B"][ep_curr, s]
                                                   + DW["W_out"][s].T)

                elif cfg["eprop_type"] == "adaptive":
                    for s in range(cfg["n_directions"]):
                        W["B"][ep_curr+ep_incr, s] = (W["W_out"][ep_curr, s]
                                                      + DW["W_out"][s]).T
                        W["B"][ep_curr+ep_incr, s] -= (
                            W["B"][ep_curr+ep_incr, s]
                            * cfg["weight_decay"])

            else:  # W, W_out, or b_out (depending on update cfg)

                W[wtype][ep_curr+ep_incr] = W[wtype][ep_curr] + DW[wtype]

                # Decay W_out
                if wtype == "W_out" and cfg["eprop_type"] == "adaptive":
                    W[wtype][ep_curr+ep_incr] -= (W[wtype][ep_curr+ep_incr]
                                                  * cfg["weight_decay"])


    if cfg["verbose"]:
        print("\nTraining complete!\n")

    # Make test batch
    randidxs = rng.integers(inps['val'].shape[0],
                            size=cfg["batch_size_test"])

    optW = ut.load_weights(log_id=log_id)

    total_testerr, perc_wrong_test, _, _ = feed_batch(
        tvt_type='test',
        cfg=cfg,
        inps=inps['val'][randidxs],
        tars=tars['val'][randidxs],
        W_rec=optW['W'],
        W_out=optW['W_out'],
        b_out=optW['b_out'],
        B=optW['B'],
        adamvars=adamvars,
        e=1,
        log_id=log_id,
        eta=eta,
        start_time=start_time)

    print(f"\nTesting complete with CE loss {total_testerr:.3f} and error "
          f"rate {100*perc_wrong_test:.1f}%!\n")

    return optVerr, perc_wrong_test


if __name__ == "__main__":

    main(cfg=CFG)
