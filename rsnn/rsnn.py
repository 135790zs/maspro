import time
import numpy as np
from config import cfg as CFG
import utils as ut
import vis


def network(cfg, inp, tar, W_rec, W_out, b_out, B):
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

            # Pad any input with zeros to make it length N_R
            Z_prev = M['Z'][t, r-1] if r > 0 else np.pad(
                M['X'][t], (0, cfg["N_R"] - len(M['X'][t])))

            M['H'][t, r] = ut.eprop_H(V=M['V'][t, r],
                                      U=M['U'][t, r],
                                      is_ALIF=M['is_ALIF'][r])

            M['ET'][t, r] = ut.eprop_ET(H=M['H'][t, r],
                                        EVV=M['EVV'][t, r],
                                        EVU=M['EVU'][t, r])

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

        M['CE'][t] = -np.sum(M['T'][t] * np.log(1e-8 + M['P'][t]))

        W = np.concatenate((
            W_rec.flatten(),
            W_out.flatten(),
            b_out))
        L2norm = np.linalg.norm(W) ** 2 * cfg["L2_reg"]

        M['DW_out'][t] = -cfg["eta"] * np.outer((M['P'][t] - M['T'][t]),
                                                M['ZbarK'][t, -1])

        L = np.dot(B, (M['P'][t] - M['T'][t]))

        # Multiply the dimensions inside the layers
        M['DW'][t] = -cfg["eta"] * np.einsum("rj,rji->rji", L, M['ETbar'][t])

        M['Db_out'][t] = -cfg["eta"] * (M['P'][t] - M['T'][t])

        # symmetric e-prop for last layer, random otherwise
        M['DB'][t, -1] = M['DW_out'][t].T

        if cfg["plot_graph"]:
            vis.plot_graph(M=M, t=t, W_rec=W_rec, W_out=W_out)
            time.sleep(0.5)

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out, B, epoch, tvt_type):
    batch_err = 0
    # TODO: Put all W of this batch in own dict
    batch_DW = {
        'DW': np.zeros(
            shape=(cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'DW_out': np.zeros(
            shape=(tars.shape[-1], cfg["N_R"],)),
        'Db_out': np.zeros(
            shape=(tars.shape[-1],)),
        'DB': np.zeros(
            shape=(cfg["N_Rec"], cfg["N_R"], tars.shape[-1],)),
    }
    for b in range(cfg["batch_size"]):
        print(f"\tEpoch {epoch}/{cfg['Epochs']-1}\t"
              f"{'  ' if tvt_type == 'val' else ''}{tvt_type} "
              f"sample {b+1}/{cfg['batch_size']}",
              end='\r' if b < cfg['Epochs']-1 else '\n')
        inps_rep = np.repeat(inps[b], cfg["Repeats"], axis=0)
        tars_rep = np.repeat(tars[b], cfg["Repeats"], axis=0)

        final_model = network(
            cfg=cfg,
            inp=inps_rep,
            tar=tars_rep,
            W_rec=W_rec,
            W_out=W_out,
            b_out=b_out,
            B=B)

        if cfg['plot_state'] and b == 0 and tvt_type == "train":
            vis.plot_state(M=final_model,
                           W_rec=W_rec,
                           W_out=W_out,
                           b_out=b_out)

        batch_err += np.sum(final_model["CE"]) / cfg["batch_size"]

        batch_DW["DW"] += np.sum(final_model['DW'], axis=0)
        batch_DW["DW_out"] += np.sum(final_model['DW_out'], axis=0)
        batch_DW["Db_out"] += np.sum(final_model['Db_out'], axis=0)

    return batch_err, batch_DW


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
    verrs = np.zeros(shape=(cfg["Epochs"]))

    optVerr = None

    W = ut.initialize_weights(tar_size=tars['train'].shape[-1])

    for e in range(0, cfg["Epochs"]):

        print(W['W'][e])
        # Make batch
        randidxs = np.random.randint(inps['train'].shape[0],
                                     size=cfg["batch_size"])
        terr, DW = feed_batch(
            epoch=e,
            tvt_type='train',
            cfg=cfg,
            inps=inps['train'][randidxs],
            tars=tars['train'][randidxs],
            W_rec=W['W'][e],
            W_out=W['W_out'][e],
            b_out=W['b_out'][e],
            B=W['B'][e])

        randidxs = np.random.randint(inps['val'].shape[0],
                                     size=cfg["batch_size"])
        verr, _ = feed_batch(
            epoch=e,
            tvt_type='val',
            cfg=cfg,
            inps=inps['val'][randidxs],
            tars=tars['val'][randidxs],
            W_rec=W['W'][e],
            W_out=W['W_out'][e],
            b_out=W['b_out'][e],
            B=W['B'][e])

        terrs[e] = terr
        verrs[e] = verr

        # Save best weights
        if optVerr is None or verr < optVerr:
            print(f"\nLowest val error ({verr:.3f}) found at epoch {e}!")
            optVerr = verr
            ut.save_weights(W=W, epoch=e)

        # Update weights for next epoch
        if not cfg["update_input_weights"]:
            DW["DW"][0, :, :inps['train'].shape[-1]] = 0
        # Update weights for next epoch
        if not cfg["update_dead_weights"]:
            DW["DW"][W["W"][e] == 0] = 0

        if e < cfg['Epochs'] - 1:
            for wtype in W.keys():
                W[wtype][e+1] = W[wtype][e] + DW[f'D{wtype}']
                W[wtype][e+1] *= cfg["weight_decay"]

        if cfg["plot_main"]:
            vis.plot_run(terrs=terrs, verrs=verrs, W=W, epoch=e)

    print("\nTraining complete!\n")

    # Make test batch
    randidxs = np.random.randint(inps['val'].shape[0],
                                 size=cfg["batch_size"])

    optW = ut.load_weights()
    total_testerr = 0

    for e in range(0, cfg["Epochs"]):
        testerr, _ = feed_batch(
            epoch=e,
            tvt_type='test',
            cfg=cfg,
            inps=inps['val'][randidxs],
            tars=tars['val'][randidxs],
            W_rec=optW['W'],
            W_out=optW['W_out'],
            b_out=optW['b_out'],
            B=optW['B'])
        total_testerr += testerr
    total_testerr /= cfg["Epochs"]

    print(f"\nTesting complete with error {total_testerr:.3f}!\n")

    return optVerr


if __name__ == "__main__":

    main(cfg=CFG)
