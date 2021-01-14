import time
import numpy as np
from config import cfg as CFG
import utils as ut
import vis


def network(cfg, inp, tar, betas, W_rec, W_out, b_out, B, adamvars, eta):

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
                                     W_rec=W_rec[s, r],
                                     betas=betas[s, r])

            M['Y'][s, t] = (
                cfg["kappa"] * M['Y'][s, t-1]
                + np.sum(W_out[s] * M['Z'][s, t, -1], axis=1)
                + b_out[s])

        # Shared output layer, so step out of subnetwork loop.
        Ysum = np.sum(M['Y'][:, t], axis=0)

        M['P'][t] = ut.eprop_P(cfg=cfg, Y=Ysum)

        M['Pmax'][t, M['P'][t].argmax()] = 1

        M['CE'][t] = -np.sum(M['T'][t] * np.log(1e-30 + M['P'][t]))
        M['Correct'][t] = int((M['Pmax'][t] == M['T'][t]).all())

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

            M['spikerate'][s, t] = np.mean(M['Z'][s, :(t+1)], axis=0)


            M['L_reg'][s, t] = (cfg["FR_reg"]
                             * (1 / n_steps)
                             * (M['spikerate'][s, t] - cfg["FR_target"]))


            for r in range(cfg["N_Rec"]):
                stdloss = ut.einsum(a=M['L_std'][s, t, r],
                                    b=M['ETbar'][s, curr_t, r])  # Checked correct
                M[f'gW'][s, curr_t, r] += stdloss
                regloss = ut.einsum(a=M['L_reg'][s, t, r],
                                    b=M['ETbar'][s, curr_t, r])
                M[f'gW'][s, curr_t, r] += regloss
                # Bellec contains error w.r.t. factoring with ET, because this can't revive silent neurons?
                # M[f'gW'][s, curr_t, r] = (M['L_reg'][s, t, r] + M[f'gW'][s, curr_t, r].T).T


            if cfg["update_W_out"]:
                M[f'gW_out'][s, curr_t] += np.outer(M['D'][t], M['Zbar'][s, t, -1])

            if cfg["update_b_out"]:
                M[f'gb_out'][s, curr_t] += M['D'][t]
        if cfg["plot_graph"]:
            vis.plot_graph(
                cfg=cfg, M=M, s=s, t=t, W_rec=W_rec, W_out=W_out)

    return M


def feed_batch(cfg, inps, tars, betas, W_rec, W_out, b_out, eta, B, tvt_type, adamvars, e, log_id, start_time=None):
    batch_err = 0
    batch_perc_wrong = 0
    batch_spikerate = np.zeros(shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"],))

    batch_gW = {
        'W': np.zeros(
            shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'W_out': np.zeros(
            shape=(cfg["n_directions"], tars.shape[-1], cfg["N_R"],)),
        'b_out': np.zeros(
            shape=(cfg["n_directions"], tars.shape[-1],)),
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

        this_inps = inps[b]
        # Crop placeholder -1's off the array.
        while np.all(this_inps[-1] == -1):
            this_inps = this_inps[:-1]

            if this_inps.size == 0:
                # Process failed: was 0 everywhere. Continue with original
                this_inps = inps[b]
                break

        # this_inps = inps[b, :this_tars.shape[0]]
        this_tars = tars[b, :this_inps.shape[0]]

        this_inps, this_tars = ut.interpolate_inputs(cfg=cfg,
                                                     inp=this_inps,
                                                     tar=this_tars,
                                                     stretch=cfg["Repeats"])

        final_model = network(
            cfg=cfg,
            inp=this_inps,
            tar=this_tars,
            betas=betas,
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
        batch_spikerate += np.mean(final_model["spikerate"], axis=1) / inps.shape[0]
        batch_err += np.mean(final_model["CE"]) / inps.shape[0]
        batch_perc_wrong += 100*np.mean(
            np.max(np.abs(final_model["Pmax"] - final_model["T"]),
                   axis=1)) / inps.shape[0]



        for w_type in ['W', 'W_out', 'b_out']:
            gw = np.sum(final_model[f'g{w_type}'], axis=1) / this_inps.shape[0]   # Checked correct
            batch_gW[w_type] += gw / inps.shape[0]  # Divide to correct for batch size?

    return batch_err, batch_perc_wrong, batch_gW, batch_spikerate


def main(cfg):

    start_time = time.time()
    rng = np.random.default_rng(seed=cfg["seed"])

    log_id = ut.get_log_id()
    ut.prepare_log(cfg=cfg, log_id=log_id)

    # Load data
    inps, tars = ut.load_data(cfg=cfg)
    n_channels = inps['train'].shape[-1]
    n_phones = tars['train'].shape[-1]

    n_train = inps['train'].shape[0]
    n_val = inps['val'].shape[0]

    print("N_train:", n_train, "N_val:", n_val)

    # Dict to track errors, eta, etc over epochs
    R = ut.initialize_tracking(cfg=cfg)

    # Weights to the network. Keys: W, W_out, b_out, B
    W = ut.initialize_weights(cfg=cfg, inp_size=n_channels, tar_size=n_phones)

    betas = ut.initialize_betas(cfg=cfg)
    # Weight update
    DW = ut.initialize_DWs(cfg=cfg, tar_size=n_phones)

    adamvars = ut.init_adam(cfg=cfg, tar_size=n_phones)

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

        if R['latest_val_err'] is None or cfg["eta_init_loss"] <= 0:
            R['eta'][e] = cfg["eta_init"]
        else:
            R['eta'][e] = min(cfg["eta_init"],
                      cfg["eta_init"] * (R['latest_val_err'] / cfg["eta_init_loss"]) ** cfg["eta_slope"])
        if e+1 < cfg["ramping"]:
            R['eta'][e] *= (e+1)/cfg["ramping"]
        if cfg["unramp"]:
            R['eta'][e] = max(0, R['eta'][e] * (cfg["unramp"] - e) / cfg["unramp"])

        # Make batch
        # randidxs = rng.integers(inps['train'].shape[0],
        #                         size=cfg["batch_size_train"])
        it = e * cfg["batch_size_train"]
        randidxs = np.arange(it, it + cfg["batch_size_train"]) % n_train

        err, perc_wrong, gW, spikerate = feed_batch(
            tvt_type='train',
            cfg=cfg,
            inps=inps['train'][randidxs],
            tars=tars['train'][randidxs],
            betas=betas,
            W_rec=W['W'][ep_curr],
            W_out=W['W_out'][ep_curr],
            b_out=W['b_out'][ep_curr],
            B=W['B'][ep_curr],
            adamvars=adamvars,
            e=e,
            log_id=log_id,
            eta=R['eta'][e],
            start_time=start_time)

        R['err']['train'][e] = err
        R[f'%wrong']['train'][e] = perc_wrong
        R['Hz'][e] = spikerate

        # Validation
        if e % cfg["val_every_E"] == 0:
            randidxs = rng.choice(inps['val'].shape[0],
                                  size=cfg["batch_size_val"],
                                  replace=False)
            err, perc_wrong, _, _ = feed_batch(
                tvt_type='val',
                cfg=cfg,
                betas=betas,
                inps=inps['val'][randidxs],
                tars=tars['val'][randidxs],
                W_rec=W['W'][ep_curr],
                W_out=W['W_out'][ep_curr],
                b_out=W['b_out'][ep_curr],
                B=W['B'][ep_curr],
                adamvars=adamvars,
                e=e,
                log_id=log_id,
                eta=R['eta'][e],
                start_time=start_time)

            R['err']['val'][e] = err
            R['latest_val_err'] = R['err']['val'][e]
            R[f'%wrong']['val'][e] = perc_wrong

            # Save best weights
            if R[f'optimal_val_err'] is None or R['err']['val'][e] < R[f'optimal_val_err']:
                if cfg["verbose"]:
                    print(f"\nLowest val error ({R['err']['val'][e]:.3f}) found at epoch {e}!\n")
                R[f'optimal_val_err'] = R['err']['val'][e]
                ut.save_weights(W=W, epoch=e if cfg["Track_weights"] else 0,
                                log_id=log_id)

            # Interpolate missing verrs
            R['err']['val'][:e+1] = ut.interpolate_verrs(R['err']['val'][:e+1])
            R[f'%wrong']['val'][:e+1] = ut.interpolate_verrs(R[f'%wrong']['val'][:e+1])

        if cfg["plot_run_interval"] and e % cfg["plot_run_interval"] == 0 and e:
            vis.plot_run(cfg=cfg, R=R, W=W, epoch=e, log_id=log_id,
                         inp_size=inps['train'].shape[-1])

        # Dont update weights in last epoch
        if e == cfg['Epochs'] - 1:
            break

        # If there's a time constraint (helpful in sweeping), break now.
        if cfg["max_duration"] and cfg["max_duration"] <= time.time() - start_time:
            break


        # Calculate DWs
        for wtype in W.keys():

            if wtype == "B" or not cfg[f"update_{wtype}"]:  # There's no DW for B. B is updated differently.
                continue

            gW[wtype] += cfg["L2_reg"] * np.linalg.norm(W[wtype][ep_curr].flatten())

            eta = (R['eta'][e] if wtype != "b_out" or cfg["eta_b_out"] is None
                   else cfg["eta_b_out"])

            if cfg["optimizer"] == "Adam":
                # Update Adam for W, W_out, b_out

                adamvars[f'm{wtype}'] = (
                    cfg["adam_beta1"] * adamvars[f'm{wtype}']
                    + (1 - cfg["adam_beta1"]) * gW[wtype])

                adamvars[f'v{wtype}'] = (
                    cfg["adam_beta2"] * adamvars[f'v{wtype}']
                    + (1 - cfg["adam_beta2"]) * (gW[wtype] ** 2))

                m = adamvars[f'm{wtype}'] / (1 - cfg["adam_beta1"] ** (e+1))
                v = adamvars[f'v{wtype}'] / (1 - cfg["adam_beta2"] ** (e+1))

                # Calculate DWs
                DW[wtype] = (-eta * (m / (np.sqrt(v) + cfg["adam_eps"])))

            elif cfg["optimizer"] == "RAdam":
                # Update Adam for W, W_out, b_out

                adamvars[f'm{wtype}'] = (
                    cfg["adam_beta1"] * adamvars[f'm{wtype}']
                    + (1 - cfg["adam_beta1"]) * gW[wtype])

                adamvars[f'v{wtype}'] = (
                    (1/cfg["adam_beta2"]) * adamvars[f'v{wtype}']
                    + (1 - cfg["adam_beta2"]) * gW[wtype] ** 2)

                m = adamvars[f'm{wtype}'] / (1 - cfg["adam_beta1"] ** (e+1))
                # v = adamvars[f'v{wtype}'] / (1 - cfg["adam_beta2"] ** (e+1))
                DW[wtype] = -eta*m

                rinf = 2/(1-cfg["adam_beta2"]) - 1
                rho = (rinf - 2 * (e+1) * cfg["adam_beta2"] ** (e+1)
                              / (1 - cfg["adam_beta1"] ** (e+1)))

                if rho > 4 and not np.any(adamvars[f'v{wtype}'] == 0):
                    alr = np.sqrt((1 - cfg["adam_beta1"] ** (e+1))
                                  / adamvars[f'v{wtype}'])
                    vrt = np.sqrt(((rho-4)*(rho-2)*rinf)
                                  / ((rinf-4)*(rinf-2)*rho))
                    DW[wtype] *= vrt * alr

                # # Calculate DWs
                # DW[wtype] = (-eta * (m / (np.sqrt(v) + cfg["adam_eps"])))

            elif cfg["optimizer"] == "SGD":
                DW[wtype] = (-eta * gW[wtype])


        if not cfg["update_input_weights"]:
            DW['W'][:, 0, :, :inps['train'].shape[-1]] = 0

        if not cfg["update_dead_weights"]:
            DW['W'][W["W"][ep_curr] == 0] = 0


        # Apply DWs
        for wtype in W.keys():

            if wtype == "B":

                if cfg["eprop_type"] == "symmetric" and cfg['update_W_out']:
                    for s in range(cfg["n_directions"]):  # TODO: Vectorize out s?
                        W["B"][ep_curr+ep_incr, s] = (W["W_out"][ep_curr, s]
                                                   + DW["W_out"][s]).T

                elif cfg["eprop_type"] == "adaptive" and cfg['update_W_out']:
                    for s in range(cfg["n_directions"]):
                        W["B"][ep_curr+ep_incr, s] = (W["B"][ep_curr, s]
                                                      + DW["W_out"][s].T)

                else:
                    W["B"][ep_curr+ep_incr] = W["B"][ep_curr]

            else:  # Normal update of W, W_out, or b_out
                W[wtype][ep_curr+ep_incr] = W[wtype][ep_curr] + DW[wtype]

        # Decay W_out
        if cfg["eprop_type"] == "adaptive":
            W["W_out"][ep_curr+ep_incr] -= (W["W_out"][ep_curr+ep_incr]
                                          * cfg["weight_decay"])
            W['B'][ep_curr+ep_incr] -= (W["B"][ep_curr+ep_incr, s]
                                        * cfg["weight_decay"])

    # Epoch loop ends here

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
        betas=betas,
        W_rec=optW['W'],
        W_out=optW['W_out'],
        b_out=optW['b_out'],
        B=optW['B'],
        adamvars=adamvars,
        e=1,
        log_id=log_id,
        eta=R['eta'][-1],
        start_time=start_time)

    print(f"\nTesting complete with CE loss {total_testerr:.3f} and error "
          f"rate {perc_wrong_test:.1f}%!\n")

    return R[f'optimal_val_err'], perc_wrong_test


if __name__ == "__main__":

    main(cfg=CFG)
