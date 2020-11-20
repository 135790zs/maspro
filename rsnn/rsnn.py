import time
import numpy as np
from config import cfg as CFG
import utils as ut
import vis


def network(cfg, inp, tar, W_rec, W_out, b_out, B, adamvars):
    n_steps = inp.shape[0]
    M = ut.initialize_model(length=n_steps, tar_size=tar.shape[-1])
    M["X"] = inp

    for t in range(n_steps):


        # Input is nonzero for first layer
        for r in range(cfg['N_Rec']):

            # Spike if V >= threshold
            M['Z'][t, r] = ut.eprop_Z(t=t,
                                      TZ=M['TZ'][r],
                                      V=M['V'][t, r],
                                      U=M['U'][t, r])
            M['TZ'][r, M['Z'][t, r]==1] = t
            # Pad any input with zeros to make it length N_R
            Z_prev = M['Z'][t, r-1] if r > 0 else np.pad(
                M['X'][t], (0, cfg["N_R"] - len(M['X'][t])))

            M['H'][t, r] = ut.eprop_H(V=M['V'][t, r],
                                      U=M['U'][t, r],
                                      is_ALIF=M['is_ALIF'][r])

            M['ET'][t, r] = ut.eprop_ET(H=M['H'][t, r],
                                        EVV=M['EVV'][t, r],
                                        EVU=M['EVU'][t, r])

            # Update weights for next epoch
            if not cfg["update_input_weights"]:
                M["EVV"][t, 0, :, :inp.shape[-1]] = 0
                M["EVU"][t, 0, :, :inp.shape[-1]] = 0
                M["ET"][t, 0, :, :inp.shape[-1]] = 0
            # Update weights for next epoch
            if not cfg["update_dead_weights"]:
                M["EVV"][t, r, W_rec[r] == 0] = 0
                M["EVU"][t, r, W_rec[r] == 0] = 0
                M["ET"][t, r, W_rec[r] == 0] = 0

            M["Z_in"][t, r] = np.concatenate((Z_prev, M['Z'][t, r]))
            M['Z_inbar'][t] = ((cfg["alpha"] * M['Z_inbar'][t-1] if t > 0 else 0)
                               + M['Z_in'][t])
            M['I'][t, r] = np.dot(W_rec[r], M["Z_in"][t, r])

            if t != n_steps - 1:
                M['EVV'][t+1, r] = ut.eprop_EVV(EVV=M['EVV'][t, r],
                                                Z_in=M["Z_in"][t, r])

                # TODO: Can do without M[ET] or M[H] or M[TZ] or M[DW].
                M['EVU'][t+1, r] = ut.eprop_EVU(Z_inbar=M['Z_inbar'][t, r],
                                                EVU=M['EVU'][t, r],
                                                H=M['H'][t, r])
                M['V'][t+1, r] = ut.eprop_V(V=M['V'][t, r],
                                            I=M['I'][t, r],
                                            Z=M['Z'][t, r])

                M['U'][t+1, r] = ut.eprop_U(U=M['U'][t, r],
                                            Z=M['Z'][t, r],
                                            is_ALIF=M['is_ALIF'][r])

        M['ETbar'][t] = ((cfg["kappa"] * M['ETbar'][t-1] if t > 0 else 0)
                         + M['ET'][t])
        M['ZbarK'][t] = ((cfg["kappa"] * M['ZbarK'][t-1] if t > 0 else 0)
                         + M['Z'][t])
        M['T'][t] = tar[t]
        M['Y'][t] = ((cfg["kappa"] * M['Y'][t-1] if t > 0 else 0)
                     + np.sum(W_out * M['Z'][t, -1], axis=1)
                     + b_out)

        ex = np.exp(M['Y'][t] - np.max(M['Y'][t]))
        M['P'][t] = ex / np.sum(ex)

        M['Pmax'][t, M['P'][t].argmax()] = 1

        # TODO: on which weights?
        W = np.concatenate((W_rec.flatten(),
                            W_out.flatten(),
                            B.flatten()))
        L2norm_W = np.linalg.norm(W) ** 2 * cfg["L2_reg"]

        M['CE'][t] = (-np.sum(M['T'][t] * np.log(1e-8 + M['P'][t]))
                      + L2norm_W)

        M['L'][t] = np.dot(B, (M['P'][t] - M['T'][t]))

        if t > 0:
            FR_reg = (
                cfg["eta"]
                * cfg["FR_reg"]
                * np.mean(np.einsum("rj,rji->rji",
                                    cfg["FR_target"] - np.mean(M['Z'][:t],
                                                               axis=0),
                                    M['ET'][t]),
                          axis=0))
        else:
            FR_reg = 0

        # Multiply the dimensions inside the layers
        M['gW'][t] = np.einsum("rj,rji->rji",
                               M['L'][t],
                               M['ETbar'][t])
        M['gW_out'][t] = np.outer((M['P'][t] - M['T'][t]),
                                   M['ZbarK'][t, -1])
        M['gb_out'][t] = M['P'][t] - M['T'][t]

        if cfg["optimizer"] == 'SGD':
            M['DW'][t] = -cfg["eta"] * M['gW'][t] + FR_reg

            M['DW_out'][t] = -cfg["eta"] * M['gW_out'][t]

            M['Db_out'][t] = -cfg["eta"] * M['gb_out'][t]

        elif cfg["optimizer"] == 'Adam':
            for wtype in ["W", "W_out", "b_out"]:
                m = (adamvars["beta1"] * adamvars[f"m{wtype}"]
                     + (1 - adamvars["beta1"]) * M[f'g{wtype}'][t])
                v = (adamvars["beta2"] * adamvars[f"v{wtype}"]
                     + (1 - adamvars["beta2"]) * M[f'g{wtype}'][t] ** 2)
                f1 = m / ( 1 - adamvars["beta1"])
                f2 = np.sqrt(v / (1 - adamvars["beta2"])) + adamvars["eps"]

                M[f'D{wtype}'][t] = cfg["eta"] * (f1 / f2)

            M['DW'][t] += FR_reg

        # symmetric e-prop for last layer, random otherwise
        if cfg["eprop_type"] == "adaptive":
            M['DB'][t, -1] = M['DW_out'][t].T

        if cfg["plot_graph"]:
            vis.plot_graph(M=M, t=t, W_rec=W_rec, W_out=W_out)
            time.sleep(0.5)

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out, B, epoch, tvt_type, adamvars, e):
    batch_err = 0

    batch_DW = {
        'W': np.zeros(
            shape=(cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'W_out': np.zeros(
            shape=(tars.shape[-1], cfg["N_R"],)),
        'b_out': np.zeros(
            shape=(tars.shape[-1],)),
        'B': np.zeros(
            shape=(cfg["N_Rec"], cfg["N_R"], tars.shape[-1],)),
    }
    batch_gW = {
        'W': np.zeros(
            shape=(cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'W_out': np.zeros(
            shape=(tars.shape[-1], cfg["N_R"],)),
        'b_out': np.zeros(
            shape=(tars.shape[-1],)),
        'B': np.zeros(
            shape=(cfg["N_Rec"], cfg["N_R"], tars.shape[-1],)),
    }
    for b in range(inps.shape[0]):
        print(f"\tEpoch {epoch}/{cfg['Epochs']-1}\t"
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

        batch_err += np.sum(final_model["CE"]) / inps.shape[0]

        for w_type in ['W', 'W_out', 'b_out']:
            batch_DW[w_type] += np.sum(final_model[f'D{w_type}'], axis=0)
            batch_gW[w_type] += np.sum(final_model[f'g{w_type}'], axis=0)

    return batch_err, batch_DW, batch_gW


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

    optVerr = None

    W = ut.initialize_weights(tar_size=tars['train'].shape[-1])
    # TODO: Put adam in W?

    adamvars = {
        'beta1': cfg['adam_beta1'],
        'beta2': cfg['adam_beta2'],
        'eps': cfg['adam_eps'],
        'mW': 0,
        'mW_out': 0,
        'mb_out': 0,
        'vW': 0,
        'vW_out': 0,
        'vb_out': 0,
    }

    for e in range(0, cfg["Epochs"]):

        # Make batch
        randidxs = np.random.randint(inps['train'].shape[0],
                                     size=cfg["batch_size_train"])
        terr, DW, gW = feed_batch(
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

        if e % cfg["val_every_E"] == 0:
            randidxs = np.random.randint(inps['val'].shape[0],
                                         size=cfg["batch_size_val"])
            verr, _, _ = feed_batch(
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


            # Save best weights
            if optVerr is None or verr < optVerr:
                print(f"\nLowest val error ({verr:.3f}) found at epoch {e}!")
                optVerr = verr
                ut.save_weights(W=W, epoch=e)

            # Interpolate missing verrs
            verrs[:e+1] = ut.interpolate_verrs(verrs[:e+1])

        # Update weights
        if e < cfg['Epochs'] - 1:

            for wtype in W.keys():
                # Update weights
                W[wtype][e+1] = W[wtype][e] + DW[wtype]

                if cfg["eprop_type"] == "adaptive" and wtype in ["W_out", "B"]:
                    W[wtype][e+1] -= cfg["weight_decay"] * W[wtype][e]

                elif cfg["eprop_type"] == "symmetric" and wtype == "B":
                    W[wtype][e+1] == (W["W_out"][e].T
                                      + DW['W_out'].T
                                      - (cfg["weight_decay"]
                                         * W["W_out"][e].T))
                # Update Adam
                if wtype != 'B':
                    adamvars[f'm{wtype}'] = (
                        adamvars["beta2"] * adamvars[f'm{wtype}']
                        + (1 - adamvars["beta2"]) * gW[wtype])
                    adamvars[f'v{wtype}'] = (
                        adamvars["beta2"] * adamvars[f'v{wtype}']
                        + (1 - adamvars["beta2"]) * gW[wtype] ** 2)



        if cfg["plot_main"]:
            vis.plot_run(terrs=terrs, verrs=verrs, W=W, epoch=e)

    print("\nTraining complete!\n")

    # Make test batch
    randidxs = np.random.randint(inps['val'].shape[0],
                                 size=cfg["batch_size_val"])

    optW = ut.load_weights()
    total_testerr = 0

    for e in range(0, cfg["Epochs"]):
        testerr, _, _ = feed_batch(
            epoch=e,
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

        total_testerr += testerr
    total_testerr /= cfg["Epochs"]

    print(f"\nTesting complete with error {total_testerr:.3f}!\n")

    return optVerr


if __name__ == "__main__":

    main(cfg=CFG)
