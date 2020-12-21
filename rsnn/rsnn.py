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

    M[f"X0"] = inp
    M[f"X1"] = np.flip(inp, axis=0)

    for t in range(n_steps-1):

        for s in range(cfg["n_directions"]):

            # Input is nonzero for first layer
            for r in range(cfg['N_Rec']):
                M = ut.process_layer(cfg=cfg, M=M, s=s, t=t, r=r, W_rec=W_rec[s, r])

            M['Y'][s, t] = ut.eprop_Y(cfg=cfg,
                                      Y=M['Y'][s, t-1] if t > 0 else 0,
                                      W_out=W_out[s],
                                      Z_last=M['Z'][s, t, -1],
                                      b_out=b_out[s])

        # Shared output layer, so step out of subnetwork loop.
        Ysum = np.sum(M['Y'][:, t], axis=0)

        M['P'][t] = ut.eprop_P(Y=Ysum)

        M['Pmax'][t, M['P'][t].argmax()] = 1

        M['CE'][t] = ut.eprop_CE(cfg=cfg,
                                  T=M['T'][t],
                                  P=M['P'][t],
                                  W_rec=W_rec,
                                  W_out=W_out,
                                  B=B)

        if cfg["Track_synapse"]:
            curr_t = t
        else:
            curr_t = 0

        # Calculate weight updates for all subnetworks
        for s in range(cfg["n_directions"]):
            M['D'][t] = M['P'][t] - M['T'][t]
            L_std = np.dot(B[s], M['D'][t])

            if t:
                rates = np.mean(M['Z'][s, :t], axis=0)
                M['spikerate'][s, t] = rates

                L_reg = (cfg["FR_reg"]
                         * (t/n_steps) * 0.5
                         * np.einsum("rj, rji -> rj",
                                     cfg["FR_target"] - rates,
                                     M["ETbar"][s, curr_t]))

            else:
                L_reg = 0

            M['L_std'][s, t] = L_std
            M['L_reg'][s, t] = L_reg

            # Calculate gradient and weight update
            # TODO: make into iterable
            for wtype in ["W", "W_out", "b_out"]:
                if not cfg["update_bias"] and wtype == "b_out":
                    continue
                if not cfg["update_W_out"] and wtype == "W_out":
                    continue


                grad = ut.eprop_gradient(wtype=wtype,
                                         D=M['D'][t],
                                         L_std=M['L_std'][s, t],
                                         L_reg=M['L_reg'][s, t],
                                         ETbar=M['ETbar'][s, curr_t],
                                         Zbar_last=M['Zbar'][s, t, -1])

                M[f'g{wtype}'][s, curr_t] += grad

                # dw = ut.eprop_DW(cfg=cfg,
                #                  gradient=grad,
                #                  wtype=wtype,
                #                  eta=eta,
                #                  adamvars=adamvars,
                #                  s=s)

                # M[f'D{wtype}'][s, curr_t] += dw



        if cfg["plot_graph"]:
            vis.plot_graph(
                cfg=cfg, M=M, s=s, t=t, W_rec=W_rec, W_out=W_out)
            time.sleep(0.5)

            # if not cfg["update_input_weights"]:
            #         M["DW"][s, curr_t, 0, :, :inp.shape[-1]] = 0

            # if not cfg["update_dead_weights"]:
            #         M["DW"][s, curr_t, W_rec[s] == 0] = 0

    print(f"\nRates: {1000*np.mean(M['spikerate']):.2f} Hz")

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out, eta, B, epoch, tvt_type, adamvars, e, log_id, start_time=None):
    batch_err = 0
    batch_perc_wrong = 0

    # TODO: Comb DW gW inits
    batch_DW = {
        'W': np.zeros(
            shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'W_out': np.zeros(
            shape=(cfg["n_directions"], tars.shape[-1], cfg["N_R"],)),
        'b_out': np.zeros(
            shape=(cfg["n_directions"], tars.shape[-1],)),
        'B': np.zeros(
            shape=(cfg["n_directions"], cfg["N_Rec"], cfg["N_R"], tars.shape[-1],)),
    }
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
                   f"Epoch {epoch}/{cfg['Epochs']-1}\t" if tvt_type != 'test'
                   else '\t'),
                  f"{'  ' if tvt_type == 'val' else ''}{tvt_type} "
                  f"sample {b+1}/{inps.shape[0]}",
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

        inps_rep = np.repeat(this_inps, cfg["Repeats"], axis=0)
        tars_rep = np.repeat(this_tars, cfg["Repeats"], axis=0)

        final_model = network(
            cfg=cfg,
            inp=inps_rep,
            tar=tars_rep,
            W_rec=W_rec,
            W_out=W_out,
            b_out=b_out,
            B=B,
            adamvars=adamvars,
            eta=eta)

        if (cfg['plot_state'] and b == 0 and tvt_type == "train"
            and cfg["plot_interval"] and e % cfg["plot_interval"] == 0):
            vis.plot_state(cfg=cfg,
                           M=final_model,
                           B=B,
                           W_rec=W_rec,
                           W_out=W_out,
                           b_out=b_out,
                           e=e,
                           log_id=log_id)

        # Aggregate the mean batch error to the aggregator.
        # Dividing by batch size to get batch mean.
        batch_err += np.mean(final_model["CE"]) / inps.shape[0]
        batch_perc_wrong += np.mean(
            np.max(np.abs(final_model["Pmax"]- final_model["T"]),
                   axis=1)) / inps.shape[0]

        for w_type in ['W', 'W_out', 'b_out']:
            # Summing and dividing because `Track_synapse' defines if dw's
            # are accumulated or listed
            gw = np.sum(final_model[f'g{w_type}'], axis=1) / inps_rep.shape[0]

            batch_gW[w_type] += gw

    if cfg["verbose"]:
        print(f"\t\tCE:      {batch_err:.3f},\n"
              f"\t\t% wrong: {100*batch_perc_wrong:.1f}%")
    return batch_err, batch_perc_wrong, batch_gW


def main(cfg):

    start_time = time.time()

    # Load data
    inps = {}
    tars = {}

    for tvt_type in cfg['n_examples'].keys():
        inps[tvt_type] = np.load(f"{cfg['wavs_fname']}_{tvt_type}_{cfg['task']}.npy")
        # Normalize [0, 1]
        inps[tvt_type] = ((inps[tvt_type] - np.min(inps[tvt_type]))
                          / np.ptp(inps[tvt_type]))
        tars[tvt_type] = np.load(f"{cfg['phns_fname']}_{tvt_type}_{cfg['task']}.npy")

    log_id = ut.get_log_id()

    ut.prepare_log(cfg=cfg, log_id=log_id)

    terrs = np.zeros(shape=(cfg["Epochs"]))
    verrs = np.ones(shape=(cfg["Epochs"])) * -1
    verr = None
    percs_wrong_t = np.zeros(shape=(cfg["Epochs"]))
    percs_wrong_v = np.ones(shape=(cfg["Epochs"])) * -1

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
        if verr is None or cfg["eta_init_loss"] <= 0:
            eta = cfg["eta_init"]
        else:
            eta = min(cfg["eta_init"],
                      cfg["eta_init"] * (verr / cfg["eta_init_loss"]) ** cfg["eta_slope"])
            print(f"Loss: {verr:.3f}, \tEta: {eta:.4f}")

        ep_curr = e if cfg["Track_weights"] else 0

        # Make batch
        randidxs = np.random.randint(inps['train'].shape[0],
                                     size=cfg["batch_size_train"])

        terr, perc_wrong_t, gW = feed_batch(
            epoch=e,
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

        if e % cfg["val_every_E"] == 0:
            randidxs = np.random.randint(inps['val'].shape[0],
                                         size=cfg["batch_size_val"])
            verr, perc_wrong_v, _ = feed_batch(
                epoch=e,
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
            percs_wrong_t[:e+1] = ut.interpolate_verrs(percs_wrong_t[:e+1])

        if cfg["plot_main"]:
            vis.plot_run(cfg=cfg, terrs=terrs, percs_wrong_t=percs_wrong_t,
                         verrs=verrs, percs_wrong_v=percs_wrong_t,
                         W=W, epoch=e, log_id=log_id, inp_size=inps['train'].shape[-1])

        if e == cfg['Epochs'] - 1:
            break

        if cfg["Track_weights"]:
            ep_curr = e
            ep_incr = 1
        else:
            ep_curr = 0
            ep_incr = 0

        if not cfg["update_input_weights"]:
            DW['W'][:, 0, :, :inp.shape[-1]] = 0

        if not cfg["update_dead_weights"]:
            DW['W'][W["W"][ep_curr] == 0] = 0

        # Calculate DWs
        for wtype in W.keys():
            if wtype == "B":
                continue

            # Update Adam for W, W_out, b_out
            adamvars[f'm{wtype}'] = (
                cfg["adam_beta1"] * adamvars[f'm{wtype}']
                + (1 - cfg["adam_beta1"]) * gW[wtype])
            adamvars[f'v{wtype}'] = (
                cfg["adam_beta2"] * adamvars[f'v{wtype}']
                + (1 - cfg["adam_beta2"]) * gW[wtype] ** 2)

            # Calculate DWs
            DW[wtype] = ((eta if wtype != "b_out" else cfg["eta_b_out"])
                         * (adamvars[f'm{wtype}']
                            / (np.sqrt(adamvars[f'v{wtype}'])
                               + cfg["adam_eps"])))

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

                if cfg["eprop_type"] in "random":
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
                if wtype == "W_out":
                    W[wtype][ep_curr+ep_incr] -= (W[wtype][ep_curr+ep_incr]
                                                  * cfg["weight_decay"])


    if cfg["verbose"]:
        print("\nTraining complete!\n")

    # Make test batch
    randidxs = np.random.randint(inps['val'].shape[0],
                                 size=cfg["batch_size_val"])

    optW = ut.load_weights(log_id=log_id)

    total_testerr, perc_wrong_test, _ = feed_batch(
        epoch=1,
        tvt_type='test',
        cfg=cfg,
        inps=inps['val'][randidxs],
        tars=tars['val'][randidxs],
        W_rec=optW['W'],
        W_out=optW['W_out'],
        b_out=optW['b_out'],
        B=optW['B'],
        adamvars=adamvars,
        e=e,
        log_id=log_id,
        eta=eta,
        start_time=start_time)

    print(f"\nTesting complete with CE loss {total_testerr:.3f} and error "
          f"rate {100*perc_wrong_test:.1f}%!\n")

    return optVerr


if __name__ == "__main__":

    main(cfg=CFG)
