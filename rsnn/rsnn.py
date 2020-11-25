import time
import numpy as np
from config import cfg as CFG
import utils as ut
import vis


def network_old(cfg, inp, tar, W_rec, W_out, b_out, B, adamvars):
    n_steps = inp.shape[0]

    M = ut.initialize_model(length=n_steps, tar_size=tar.shape[-1])
    M["X"] = inp
    M["T"] = tar

    for t in range(n_steps):

        # Input is nonzero for first layer
        for r in range(cfg['N_Rec']):
            M = ut.process_layer(M=M, t=t, r=r, W_rec=W_rec[r])


        M['Y'][t] = ut.eprop_Y(Y=M['Y'][t-1] if t > 0 else 0,
                               W_out=W_out,
                               Z_last=M['Z'][t, -1],
                               b_out=b_out)

        M['P'][t] = ut.eprop_P(Y=M['Y'][t])

        M['Pmax'][t, M['P'][t].argmax()] = 1

        M['CE'][t] = ut.eprop_CE(T=M['T'][t],
                                 P=M['P'][t],
                                 W_rec=W_rec,
                                 W_out=W_out,
                                 B=B)

        M['L'][t] = np.dot(B, (M['P'][t] - M['T'][t]))

        # Calculate gradient and weight update
        # TODO: make into iterable
        for wtype in ["W", "W_out", "b_out"]:
            if not cfg["update_bias"] and wtype == "b_out":
                continue
            if not cfg["update_W_out"] and wtype == "W_out":
                continue
            M[f'g{wtype}'][t] = ut.eprop_gradient(wtype=wtype,
                                                  L=M['L'][t],
                                                  ETbar=M['ETbar'][t],
                                                  Zbar_last=M['ZbarK'][t, -1],
                                                  P=M['P'][t],
                                                  T=M['T'][t])


            M[f'D{wtype}'][t] = ut.eprop_DW(wtype=wtype,
                                            adamvars=adamvars,
                                            gradient=M[f'g{wtype}'][t],
                                            Zs=M['Z'][:t],
                                            ET=M['ET'][t])

        if not cfg["update_input_weights"]:
                M["DW"][t, 0, :, :inp.shape[-1]] = 0

        if not cfg["update_dead_weights"]:
                M["DW"][t, W_rec == 0] = 0

        # symmetric e-prop for last layer, random otherwise
        if cfg["eprop_type"] == "adaptive":
            M['DB'][t, -1] = M['DW_out'][t].T

        if cfg["plot_graph"]:
            vis.plot_graph(
                M=M, t=t, W_rec=W_rec, W_out=W_out)
            time.sleep(0.5)

    return M


def network(cfg, inp, tar, W_rec, W_out, b_out, B, adamvars):
    n_steps = inp.shape[0]

    M = ut.initialize_model(length=n_steps, tar_size=tar.shape[-1])
    M["T"] = tar

    for s in range(cfg["n_directions"]):
        M["X"] = inp if s == 0 else np.flip(inp, axis=0)

        for t in range(n_steps):

            # Input is nonzero for first layer
            for r in range(cfg['N_Rec']):
                M = ut.process_layer(M=M, s=s, t=t, r=r, W_rec=W_rec[s, r])


            M['Y'][s, t] = ut.eprop_Y(Y=M['Y'][s, t-1] if t > 0 else 0,
                                      W_out=W_out[s],
                                      Z_last=M['Z'][s, t, -1],
                                      b_out=b_out[s])
        if s == 1:
            M['Y'][1] = np.flip(M['Y'][1], axis=0)

        for t in range(n_steps):  # TODO: can make more efficient
            M['P'][t] = ut.eprop_P(Y=np.sum(M['Y'][:, t], axis=0))

            M['Pmax'][t, M['P'][t].argmax()] = 1

            M['CE'][t] = ut.eprop_CE(T=M['T'][t],
                                     P=M['P'][t],
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


                    M[f'D{wtype}'][s, t] = ut.eprop_DW(wtype=wtype,
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
                        M=M, s=s, t=t, W_rec=W_rec, W_out=W_out)
                    time.sleep(0.5)

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out, B, epoch, tvt_type, adamvars, e):
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
        print((f"\tEpoch {epoch}/{cfg['Epochs']-1}\t" if tvt_type != 'test'
               else '\t'),
              f"{'  ' if tvt_type == 'val' else ''}{tvt_type} "
              f"sample {b+1}/{inps.shape[0]}",
              end='\r' if b < inps.shape[0]-1 else '\n')

        inps_rep = np.repeat(inps[b], cfg["Repeats"], axis=0)
        tars_rep = np.repeat(tars[b], cfg["Repeats"], axis=0)

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
            vis.plot_state(M=final_model,
                           W_rec=W_rec,
                           W_out=W_out,
                           b_out=b_out)

        batch_err += np.sum(final_model["CE"]) / inps.shape[0]  # TODO: use mean over axis
        batch_perc_wrong += np.mean(
            np.max(np.abs(final_model["Pmax"]- final_model["T"]),
                   axis=1)) / inps.shape[0]

        for w_type in ['W', 'W_out', 'b_out']:
            batch_DW[w_type] += np.mean(final_model[f'D{w_type}'], axis=1)
            batch_gW[w_type] += np.mean(final_model[f'g{w_type}'], axis=1)

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

    terrs = np.zeros(shape=(cfg["Epochs"]))
    verrs = np.ones(shape=(cfg["Epochs"])) * -1
    percs_wrong_t = np.zeros(shape=(cfg["Epochs"]))
    percs_wrong_v = np.ones(shape=(cfg["Epochs"])) * -1

    optVerr = None

    W = ut.initialize_weights(tar_size=tars['train'].shape[-1])
    # TODO: Put adam in W?

    adamvars = ut.init_adam(tar_size=tars['train'].shape[-1])

    for e in range(0, cfg["Epochs"]):
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
            e=e)
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
                e=e)
            verrs[e] = verr
            percs_wrong_v[e] = perc_wrong_v


            # Save best weights
            if optVerr is None or verr < optVerr:
                print(f"\nLowest val error ({verr:.3f}) found at epoch {e}!\n")
                optVerr = verr
                ut.save_weights(W=W, epoch=e)

            # Interpolate missing verrs
            verrs[:e+1] = ut.interpolate_verrs(verrs[:e+1])
            percs_wrong_t[:e+1] = ut.interpolate_verrs(percs_wrong_t[:e+1])

        if cfg["plot_main"]:
            vis.plot_run(terrs=terrs, percs_wrong_t=percs_wrong_t,
                         verrs=verrs, percs_wrong_v=percs_wrong_t,
                         W=W, epoch=e)

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

    print("\nTraining complete!\n")

    # Make test batch
    randidxs = np.random.randint(inps['val'].shape[0],
                                 size=cfg["batch_size_val"])

    optW = ut.load_weights()

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
        e=e)

    print(f"\nTesting complete with CE loss {total_testerr:.3f} and error "
          f"rate {100*perc_wrong_test:.1f}%!\n")

    return optVerr


if __name__ == "__main__":

    main(cfg=CFG)
