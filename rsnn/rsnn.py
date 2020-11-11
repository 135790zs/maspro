import time
import numpy as np
from config import cfg as CFG
import utils as ut
import vis


def network(cfg, inp, tar, W_rec, W_out, b_out):
    n_steps = inp.shape[0]
    M = ut.initialize_model(length=n_steps)
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
            M['I'][t, r] = np.dot(W_rec[r], M["Z_in"][t, r])

            if t != n_steps - 1:
                M['EVV'][t+1, r] = ut.eprop_EVV(EVV=M['EVV'][t, r],
                                                Z_in=M["Z_in"][t, r])

                # TODO: Can do without M[ET] or M[H] or M[TZ] or M[DW].
                M['EVU'][t+1, r] = ut.eprop_EVU(EVV=M['EVV'][t, r],
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
        # print(np.sum(W_out * M['Z'][t, -1], axis=0))
        mx = np.max(M['Y'][t])
        ex = np.exp(M['Y'][t] / (mx + 1e-8))
        M['P'][t] = ex / np.sum(ex)

        # print(M['P'][t].argmax())
        M['Pmax'][t, M['P'][t].argmax()] = 1

        M['E'][t] = - np.sum(M['T'][t] * np.log(M['P'][t]))

        M['DW_out'][t] = -cfg["eta"] * M['ZbarK'][t] * np.sum(
            M['P'][t] - M['T'][t])
        B = M['DW_out'][t].T
        intersum = np.sum(B * (M['P'][t] - M['T'][t]), axis=0)
        M['DW'][t] = -cfg["eta"] * M['ETbar'][t] * np.sum(
            intersum)
        M['Db_out'][t] = -cfg["eta"] * np.sum(M['P'][t] - M['T'][t])

        if cfg["plot_graph"]:
            vis.plot_graph(M=M, t=t, W_rec=W_rec, W_out=W_out)
            time.sleep(0.5)

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out, epoch, tvt_type):
    batch_err = 0
    # TODO: Put all W of this batch in own dict
    batch_DW = {
        'DW': np.zeros(
            shape=(cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'DW_out': np.zeros(
            shape=(cfg["N_O"], cfg["N_R"],)),
        'Db_out': np.zeros(
            shape=(cfg["N_O"],)),
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
            b_out=b_out)

        if cfg['plot_state'] and epoch == 0 and b == 0 and tvt_type == "train":
            vis.plot_state(M=final_model,
                           W_rec=W_rec,
                           W_out=W_out,
                           b_out=b_out)

        batch_err += ut.get_error(
            M=final_model,
            tars=tars_rep,
            W_out=W_out,
            b_out=b_out)

        batch_DW["DW"] += np.sum(final_model['DW'], axis=0)
        batch_DW["DW_out"] += np.sum(final_model['DW_out'], axis=0)
        batch_DW["Db_out"] += np.sum(final_model['Db_out'], axis=0)


    batch_loss = ut.get_loss(err=batch_err)

    return batch_loss, batch_DW


def main(cfg):
    W = ut.initialize_weights()
    optVerr = None
    terrs = np.zeros(shape=(cfg["Epochs"]))
    verrs = np.zeros(shape=(cfg["Epochs"]))

    # Load data
    inps = {}
    tars = {}
    for tvt_type in cfg['n_examples'].keys():
        inps[tvt_type] = np.load(f'{cfg["wavs_fname"]}_{tvt_type}.npy')
        # Normalize [0, 1]
        inps[tvt_type] = ((inps[tvt_type] - np.min(inps[tvt_type]))
                          / np.ptp(inps[tvt_type]))
        tars[tvt_type] = np.load(f'{cfg["phns_fname"]}_{tvt_type}.npy')

    for e in range(0, cfg["Epochs"]):

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
            b_out=W['b_out'][e])

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
            b_out=W['b_out'][e],)

        terrs[e] = terr
        verrs[e] = verr

        # Save best weights
        if optVerr is None or verr < optVerr:
            print(f"\nLowest val error ({verr:.3f}) found at epoch {e}!")
            optVerr = verr
            ut.save_weights(W=W, epoch=e)

        # Update weights for next epoch
        if e < cfg['Epochs'] - 1:
            W['W'][e+1] = W['W'][e] + DW['DW']
            W['W'][e+1] *= cfg["weight_decay"]
            W['W_out'][e+1] = W['W_out'][e] + DW['DW_out']
            W['W_out'][e+1] *= cfg["weight_decay"]
            W['b_out'][e+1] = W['b_out'][e] + DW['Db_out']
            W['b_out'][e+1] *= cfg["weight_decay"]

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
            b_out=optW['b_out'])
        total_testerr += testerr
    total_testerr /= cfg["Epochs"]

    print(f"\nTesting complete with error {total_testerr:.3f}!\n")

    return optVerr


if __name__ == "__main__":

    main(cfg=CFG)
