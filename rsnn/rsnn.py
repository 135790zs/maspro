import numpy as np
from config import cfg as CFG
import utils as ut
from tqdm import tqdm
import time
import vis
from task import narma10


def network(cfg, inp, W_rec, W_out):
    n_steps = inp.shape[0]
    M = ut.initialize_model(length=n_steps)
    M["XZ"] = np.random.default_rng().binomial(n=1, p=inp)

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
                M['XZ'][t], (0, cfg["N_R"] - len(M['XZ'][t])))

            M['H'][t, r] = ut.eprop_H(t=t,
                                      V=M['V'][t, r],
                                      U=M['U'][t, r],
                                      is_ALIF=M['is_ALIF'][r])

            M['ET'][t, r] = ut.eprop_ET(H=M['H'][t, r],
                                        EVV=M['EVV'][t, r],
                                        EVU=M['EVU'][t, r],
                                        is_ALIF=M['is_ALIF'][r])

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
                                            U=M['U'][t, r],
                                            I=M['I'][t, r],
                                            Z=M['Z'][t, r])

                M['U'][t+1, r] = ut.eprop_U(U=M['U'][t, r],
                                            V=M['V'][t, r],
                                            Z=M['Z'][t, r],
                                            is_ALIF=M['is_ALIF'][r])  # Aggregate weights only

        if cfg["plot_graph"]:
            vis.plot_graph(M=M, t=t, W_rec=W_rec, W_out=W_out)
            time.sleep(0.5)

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out, epoch, train):
    batch_err = 0
    DW = {
        'DW': np.zeros(
            shape=(cfg["N_Rec"], cfg["N_R"], cfg["N_R"] * 2,)),
        'DW_out': np.zeros(
            shape=(cfg["N_O"], cfg["N_R"],)),
        'Db_out': np.zeros(
            shape=(cfg["N_O"],)),
    }
    for b in range(cfg["batch_size"]):
        print(f"Epoch {epoch}/{cfg['Epochs']-1}\t"
              f"{'  Training' if train else 'Validation'} "
              f"sample {b}/{cfg['batch_size']-1}", end='\r')
        inps_rep = np.repeat(inps[b], cfg["Repeats"], axis=0)
        tars_rep = np.repeat(tars[b], cfg["Repeats"], axis=0)

        final_model = network(
            cfg=cfg, 
            inp=inps_rep, 
            W_rec=W_rec,
            W_out=W_out)

        if cfg['plot_state'] and epoch == 0 and train:
            vis.plot_state(M=final_model)

        batch_err += ut.get_error(
            M=final_model,
            tars=tars_rep,
            cfg=cfg,
            W_out=W_out,
            b_out=b_out)
    
        DW = ut.update_DWs(
            cfg=cfg,
            DW=DW,
            err=batch_err,
            M=final_model)

    batch_loss = ut.get_loss(err=batch_err)

    return batch_loss, DW


def main(cfg):
    rng = np.random.default_rng()
    W = ut.initialize_weights()
    optW = None
    optVerr = None
    terrs = np.zeros(shape=(cfg["Epochs"]))
    verrs = np.zeros(shape=(cfg["Epochs"]))

    # Load data
    inps = np.load(cfg["wavs_fname"])
    tars = np.load(cfg["phns_fname"])

    # Normalize [0, 1]
    inps = (inps - np.min(inps)) / np.ptp(inps)

    for e in range(0, cfg["Epochs"]):

        # Make batch
        randidxs = np.random.randint(inps.shape[0], size=cfg["batch_size"])
        terr, DW = feed_batch(
            epoch=e,
            train=True,
            cfg=cfg,
            inps=inps[randidxs],
            tars=tars[randidxs],
            W_rec=W['W'][e],
            W_out=W['W_out'][e],
            b_out=W['b_out'][e])

        # Update weights for next epoch
        W['W'][e+1] = W['W'][e] + DW['DW']
        W['W'][e+1] *= cfg["weight_decay"]
        W['W_out'][e+1] = W['W_out'][e] + DW['DW_out']
        W['W_out'][e+1] *= cfg["weight_decay"]
        W['b_out'][e+1] = W['b_out'][e] + DW['Db_out']
        W['b_out'][e+1] *= cfg["weight_decay"]


        # TODO: change to actual val data
        randidxs = np.random.randint(inps.shape[0], size=cfg["batch_size"])
        verr, _ = feed_batch(
            epoch=e,
            train=False,
            cfg=cfg,
            inps=inps[randidxs],
            tars=tars[randidxs],
            W_rec=W['W'][e],
            W_out=W['W_out'][e],
            b_out=W['b_out'][e],)

        # Save best weights
        if optVerr is None or verr < optVerr:
            print(f"\nLowest error ({verr:.3f}) found at epoch {e}!")
            optVerr = verr
            optW = W

        terrs[e] = terr
        verrs[e] = verr

        if cfg["plot_main"]:
            vis.plot_run(terrs=terrs, verrs=verrs, W=W, epoch=e)

    print("\nTraining complete!")

    return optVerr


if __name__ == "__main__":

    print(main(cfg=CFG))
