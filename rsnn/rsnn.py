import numpy as np
from config import cfg as CFG
import utils as ut
import vis
from task import narma10


def run_network(cfg, M, W, e, d):
    """ Run a LSNN using vars `M' and `W' for `repeats' steps, offset by `d'.
    """

    for t0 in range(cfg["Repeats"]):
        t = t0 + d * cfg['Repeats']  # Continue in M where we left off from the previous frame
        print(f"\t\tStep {t+1}/{cfg['Repeats'] * cfg['batch_size']}")

        # Input is nonzero for first layer
        for r in range(cfg['N_Rec']):

            # Spike if V >= threshold
            M['Z'][t, r] = ut.eprop_Z(t=t,
                                      TZ=M['TZ'][r],
                                      V=M['V'][t, r],
                                      U=M['U'][t, r])

            # Pad any input with zeros to make it length N_R
            Z_prev = M['Z'][t, r-1] if r > 0 else np.pad(
                M['XZ'][t], (0, cfg["N_R"]-len(M['XZ'][t])))

            M['H'][t, r] = ut.eprop_H(t=t,
                                      V=M['V'][t, r],
                                      U=M['U'][t, r],
                                      is_ALIF=M['is_ALIF'][r])

            M['ET'][t, r] = ut.eprop_ET(H=M['H'][t, r],
                                        EVV=M['EVV'][t, r],
                                        EVU=M['EVU'][t, r],
                                        is_ALIF=M['is_ALIF'][r])

            M["Z_in"][t, r] = np.concatenate((Z_prev, M['Z'][t, r]))
            M['I'][t, r] = np.dot(W['W'][e, r], M["Z_in"][t, r])

            if t != cfg["Repeats"] * cfg["batch_size"] - 1:
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
            # print(M['EVV'][t, r].shape)
            if t > 0:
                M['ETbar'][t, r] = cfg["kappa"] * M["ETbar"][t-1, r] + M["ET"][t, r]

            # if t != cfg["Repeats"]*cfg['batch_size'] - 1:  # TODO: WHY??

            # Calculate network output
            M['Y'][t] = (cfg["kappa"] * M['Y'][t-1]
                         + np.sum(W['W_out'][e-1] * M['Z'][t, -1])
                         + W['b_out'][e-1])

            if cfg["task"] == "narma10":
                M['T'][t] = narma10(t=t, u=M['X'][:t], y=M['Y'][:t])
            elif cfg["task"] in ["sinusoid", "pulse"]:  # TODO: Can delegate to init
                M["T"][t] = np.mean(M["X"][t])

            # For some tasks, the desired output is the source of the input
            if cfg["tasktype"] == "regression":
                M["error"][t] = (M["Y"][t] - M["T"][t])
                M["loss"][t] = np.sum(M["error"][t]**2)
            elif cfg["tasktype"] == "classification":
                sm = np.exp(M['Y'][t]) / np.sum(np.exp(M['Y'][t]))
                M["error"][t] = (sm - M["T"][t])
                M["loss"][t] = -np.sum(M['T'][t] * np.log(sm))

            print(f"\t\tError: {M['error'][t]}")

            M['DW'][t] = M['DW'][t-1] - cfg["eta"] * np.sum(
                W['B'][e-1] * M["error"][t]) * ut.temporal_filter(
                    cfg["kappa"], M['ET'][:t+1])

            # freeze input weights
            M['DW'][t, 0, :, :cfg["N_R"]].fill(0.)
            M["DW_out"][t] = -cfg["eta"] * np.sum(
                M["error"][t]) * ut.temporal_filter(
                    cfg["kappa"], M['Z'][:t+1, -1])

            M["Db_out"][t] = -cfg["eta"] * np.sum(M["error"][t])

        if (t > 0
                and cfg["plot_interval"]
                and ((t+1) % cfg["plot_interval"] == 0)):
            vis.plot_drsnn(M=M,
                           t=t,
                           layers=None,   # Tuple for 2, None for heatmap
                           neurons=None)  # Idem  # Move example through network for t steps (5 ms)
    return M, W


def network(cfg, inp, W_rec):
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
                M['XZ'][t], (0, cfg["N_R"]-len(M['XZ'][t])))

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

    return M


def feed_batch(cfg, inps, tars, W_rec, W_out, b_out):
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
        inps_rep = np.repeat(inps[b], cfg["Repeats"], axis=0)
        tars_rep = np.repeat(tars[b], cfg["Repeats"], axis=0)

        final_model = network(cfg=cfg, inp=inps_rep, W_rec=W_rec)
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

    return batch_err, DW


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

    for e in range(0, cfg["Epochs"]):

        # Make batch
        randidxs = np.random.randint(42, size=cfg["batch_size"])
        terr, DW = feed_batch(
            cfg=cfg,
            inps=inps[randidxs],
            tars=tars[randidxs],
            W_rec=W['W'][e],
            W_out=W['W_out'][e],
            b_out=W['b_out'][e])

        W = ut.update_weight(cfg=cfg, W=W, DW=DW)

        # TODO: change to actual val data
        randidxs = np.random.randint(42, size=cfg["batch_size"])
        verr = feed_batch(
            cfg=cfg,
            inps=inps[randidxs],
            tars=tars[randidxs],
            W_rec=W['W'][e])

        # Save best weights
        if optVerr is None or verr < optVerr:
            optVerr = verr
            optW = W

        terrs[e] = terr
        verrs[e] = verr

        vis.plot_error(terrs, verrs)


def run_rsnn(cfg):

    # Initialize weights
    rng = np.random.default_rng()
    W = ut.initialize_weights()
    inps = np.load(cfg["wavs_fname"])
    tars = np.load(cfg["phns_fname"])
    inps = (inps - np.min(inps)) / np.ptp(inps)  # Scale [0, 1]

    for e in range(1, cfg["Epochs"]):  # Move e batches through network (80)
        print(f"Epoch {e}/{cfg['Epochs']}")

        print(f"\tMaking random batch...")
        # Get random batch
        ridx = np.random.randint(42, size=cfg["batch_size"])
        b_inps = inps[ridx]
        b_tars = tars[ridx]
        M = ut.initialize_log()  # Except weights

        for d in range(cfg["batch_size"]):  # Move d examples through network (32)
            print(f"\tBatch frame {d+1}/{cfg['batch_size']}")

            M['X'] = b_inps[d]
            M['T'] = b_tars[d]
            M['X'] = np.repeat(M['X'], cfg["Repeats"], axis=0)
            M['T'] = np.repeat(M['T'], cfg["Repeats"], axis=0)

            M["XZ"] = rng.binomial(n=1, p=M["X"])

            M, W = run_network(cfg=cfg, M=M, W=W, e=e, d=d)

        # Update weights
        print(f"\tUpdating weights...")
        if cfg["update_dead_weights"]:
            for r2 in range(cfg["N_Rec"]):
                # Zero diag E: no self-conn
                for t in range(cfg["Repeats"]):
                    np.fill_diagonal(M['DW'][t, r2, :, cfg["N_R"]:], 0)
        else:
            # Don't update zero-weights
            M['DW'] = np.where(W['W'], M['DW'], 0.)

        W["B"][e] = W["B"][e-1] + np.sum(np.transpose(M["DW_out"], axes=(0, 2, 1)), axis=0)
        W["B"][e] *= cfg["weight_decay"]
        W['W'][e] = W['W'][e-1] + np.sum(M['DW'], axis=0)
        W["W_out"][e] = W['W_out'][t] + np.sum(M['DW_out'], axis=0)
        W["W_out"][e] *= cfg["weight_decay"]

        W["b_out"][e] = W["b_out"][e-1] * np.sum(M["Db_out"], axis=0)

        # Test on validation set

    # Return lowest validation error

    return np.mean(M["loss"])


if __name__ == "__main__":

    print(main(cfg=CFG))
