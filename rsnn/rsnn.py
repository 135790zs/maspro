import time
import numpy as np
from config import cfg as CFG
import utils as ut
import vis


def network(cfg, inp, tar, W_rec, W_out, b_out, B, adamvars):
    inp = np.pad(array=inp, mode='edge', pad_width=((0, cfg["delay"]), (0, 0)))

    n_steps = inp.shape[0]
    M = ut.initialize_model(cfg=cfg, length=n_steps, tar_size=tar.shape[-1])

    M["T"] = tar
    M["T"] = np.pad(array=M["T"], mode='edge', pad_width=((cfg["delay"], 0), (0, 0)))

    for s in range(cfg["n_directions"]):
        M[f"X{s}"] = inp if s == 0 else np.flip(inp, axis=0)

        for t in range(n_steps):

            # Input is nonzero for first layer
            for r in range(cfg['N_Rec']):
                M = ut.process_layer(cfg=cfg, M=M, s=s, t=t, r=r, W_rec=W_rec[s, r])


            M['Y'][s, t] = ut.eprop_Y(cfg=cfg,
                                      Y=M['Y'][s, t-1] if t > 0 else 0,
                                      W_out=W_out[s],
                                      Z_last=M['Z'][s, t, -1],
                                      b_out=b_out[s])

        Ysum = M['Y'][0] + (np.flip(M['Y'][1], axis=0)
                            if cfg["n_directions"] > 1 else 0)

    M['P'] = ut.eprop_P(Y=Ysum)
    M['Pmax'][range(M['P'].shape[0]),
              M['P'].argmax(axis=1)] = 1

    M['CE'] = ut.eprop_CE(cfg=cfg,
                          T=M['T'],
                          P=M['P'],
                          W_rec=W_rec,
                          W_out=W_out,
                          B=B)

    for s in range(cfg["n_directions"]):

        for t in range(n_steps):  # TODO: can make more efficient by doing einsums
            M['L'][s, t] = np.dot(B[s], (M['P'][t] - M['T'][t]))

            # Calculate gradient and weight update
            # TODO: make into iterable
            for wtype in ["W", "W_out", "b_out"]:
                if not cfg["update_bias"] and wtype == "b_out":
                    continue
                if not cfg["update_W_out"] and wtype == "W_out":
                    continue

                M[f'g{wtype}'][s, t] = ut.eprop_gradient(wtype=wtype,
                                                         L=M['L'][s, t],
                                                         ETbar=M['ETbar'][s, t],
                                                         Zbar_last=M['ZbarK'][s, t, -1],
                                                         P=M['P'][t],
                                                         T=M['T'][t])


                M[f'D{wtype}'][s, t] = ut.eprop_DW(cfg=cfg,
                                                   wtype=wtype,
                                                   s=s,
                                                   adamvars=adamvars,
                                                   gradient=M[f'g{wtype}'][s, t],
                                                   Zs=M['Z'][s, :t],
                                                   ET=M['ET'][s, t])

            if not cfg["update_input_weights"]:
                    M["DW"][s, t, 0, :, :inp.shape[-1]] = 0

            if not cfg["update_dead_weights"]:
                    M["DW"][s, t, W_rec[s] == 0] = 0

            # symmetric e-prop for last layer, random otherwise
            if cfg["eprop_type"] == "adaptive":
                M['DB'][s, t, -1] = M['DW_out'][s, t].T

            if cfg["plot_graph"]:
                vis.plot_graph(
                    cfg=cfg, M=M, s=s, t=t, W_rec=W_rec, W_out=W_out)
                time.sleep(0.5)

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out, B, epoch, tvt_type, adamvars, e, log_id):
    batch_err = 0
    batch_perc_wrong = 0

    # Comb DW gW inits
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
        if cfg["verbose"]:
            print((f"({log_id})\tEpoch {epoch}/{cfg['Epochs']-1}\t" if tvt_type != 'test'
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
            adamvars=adamvars)

        if (cfg['plot_state'] and b == 0 and tvt_type == "train"
            and cfg["plot_interval"] and e % cfg["plot_interval"] == 0):
            vis.plot_state(cfg=cfg,
                           M=final_model,
                           W_rec=W_rec,
                           W_out=W_out,
                           b_out=b_out,
                           e=e,
                           log_id=log_id)

        batch_err += np.sum(final_model["CE"]) / inps.shape[0]  # TODO: use mean over axis
        batch_err += np.sum(final_model["CE"]) / inps.shape[0]  # TODO: use mean over axis
        batch_perc_wrong += np.mean(
            np.max(np.abs(final_model["Pmax"]- final_model["T"]),
                   axis=1)) / inps.shape[0]

        for w_type in ['W', 'W_out', 'b_out']:
            batch_DW[w_type] += np.mean(final_model[f'D{w_type}'], axis=1)
            batch_gW[w_type] += np.mean(final_model[f'g{w_type}'], axis=1)

    if cfg["verbose"]:
        print(f"\t\tCE:      {batch_err:.3f},\n"
              f"\t\t% wrong: {100*batch_perc_wrong:.1f}%")
    return batch_err, batch_perc_wrong, batch_DW, batch_gW


def main(cfg):
    # Load data
    inps = {}
    tars = {}
    for tvt_type in cfg['n_examples'].keys():
        inps[tvt_type] = np.load(f'{cfg["wavs_fname"]}_{tvt_type}.npy')
        # Normalize [0, 1]
        inps[tvt_type] = ((inps[tvt_type] - np.min(inps[tvt_type]))
                          / np.ptp(inps[tvt_type]))
        tars[tvt_type] = np.load(f'{cfg["phns_fname"]}_{tvt_type}.npy')

    log_id = ut.get_log_id()

    ut.prepare_log(cfg=cfg, log_id=log_id)

    terrs = np.zeros(shape=(cfg["Epochs"]))
    verrs = np.ones(shape=(cfg["Epochs"])) * -1
    percs_wrong_t = np.zeros(shape=(cfg["Epochs"]))
    percs_wrong_v = np.ones(shape=(cfg["Epochs"])) * -1

    optVerr = None

    W = ut.initialize_weights(cfg=cfg, tar_size=tars['train'].shape[-1])
    # TODO: Put adam in W?

    adamvars = ut.init_adam(cfg=cfg, tar_size=tars['train'].shape[-1])

    for e in range(0, cfg["Epochs"]):
        if not cfg["verbose"]:
            print(f"ep {e}/{cfg['Epochs']}", end='\r')
        # Make batch
        randidxs = np.random.randint(inps['train'].shape[0],
                                     size=cfg["batch_size_train"])
        terr, perc_wrong_t, DW, gW = feed_batch(
            epoch=e,
            tvt_type='train',
            cfg=cfg,
            inps=inps['train'][randidxs],
            tars=tars['train'][randidxs],
            W_rec=W['W'][e],
            W_out=W['W_out'][e],
            b_out=W['b_out'][e],
            B=W['B'][e],
            adamvars=adamvars,
            e=e,
            log_id=log_id)
        terrs[e] = terr
        percs_wrong_t[e] = perc_wrong_t

        if e % cfg["val_every_E"] == 0:
            randidxs = np.random.randint(inps['val'].shape[0],
                                         size=cfg["batch_size_val"])
            verr, perc_wrong_v, _, _ = feed_batch(
                epoch=e,
                tvt_type='val',
                cfg=cfg,
                inps=inps['val'][randidxs],
                tars=tars['val'][randidxs],
                W_rec=W['W'][e],
                W_out=W['W_out'][e],
                b_out=W['b_out'][e],
                B=W['B'][e],
                adamvars=adamvars,
                e=e,
                log_id=log_id)
            verrs[e] = verr
            percs_wrong_v[e] = perc_wrong_v


            # Save best weights
            if optVerr is None or verr < optVerr:
                if cfg["verbose"]:
                    print(f"\nLowest val error ({verr:.3f}) found at epoch {e}!\n")
                optVerr = verr
                ut.save_weights(W=W, epoch=e, log_id=log_id)

            # Interpolate missing verrs
            verrs[:e+1] = ut.interpolate_verrs(verrs[:e+1])
            percs_wrong_t[:e+1] = ut.interpolate_verrs(percs_wrong_t[:e+1])

        if cfg["plot_main"]:
            vis.plot_run(terrs=terrs, percs_wrong_t=percs_wrong_t,
                         verrs=verrs, percs_wrong_v=percs_wrong_t,
                         W=W, epoch=e, log_id=log_id)

        if e == cfg['Epochs'] - 1:
            break

        # Update weights
        for wtype in W.keys():
            # Update weights
            W[wtype][e+1] = W[wtype][e] + DW[wtype]

            if not cfg["update_bias"] and wtype == "b_out":
                continue
            if not cfg["update_W_out"] and wtype == "W_out":
                continue

            # Decay iff adaptive
            if cfg["eprop_type"] == "adaptive" and wtype in ["W_out", "B"]:
                W[wtype][e+1] -= cfg["weight_decay"] * W[wtype][e+1]

            # Mirror B <-> W_out if symmetric
            elif cfg["eprop_type"] == "symmetric" and wtype == "B":
                for s in range(cfg["n_directions"]):
                    W[wtype][e+1, s] = (W["W_out"][e, s].T
                                        + DW['W_out'][s].T
                                        - (cfg["weight_decay"]
                                           * W["W_out"][e, s].T))
            # Update Adam
            if wtype != 'B':
                adamvars[f'm{wtype}'] = (
                    adamvars["beta2"] * adamvars[f'm{wtype}']
                    + (1 - adamvars["beta2"]) * gW[wtype])
                adamvars[f'v{wtype}'] = (
                    adamvars["beta2"] * adamvars[f'v{wtype}']
                    + (1 - adamvars["beta2"]) * gW[wtype] ** 2)

    if cfg["verbose"]:
        print("\nTraining complete!\n")

    # Make test batch
    randidxs = np.random.randint(inps['val'].shape[0],
                                 size=cfg["batch_size_val"])

    optW = ut.load_weights(log_id=log_id)

    total_testerr, perc_wrong_test, _, _ = feed_batch(
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
        log_id=log_id)

    print(f"\nTesting complete with CE loss {total_testerr:.3f} and error "
          f"rate {100*perc_wrong_test:.1f}%!\n")

    return optVerr


if __name__ == "__main__":

    main(cfg=CFG)
