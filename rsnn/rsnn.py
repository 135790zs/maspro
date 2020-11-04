import numpy as np
from config import cfg as CFG
import utils as ut
import vis
from task import narma10


def run_rsnn(cfg):

    # Initialize weights
    rng = np.random.default_rng()
    W = ut.initialize_weights()
    inps = np.load(cfg["wavs_fname"])
    tars = np.load(cfg["phns_fname"])
    inps = (inps - np.min(inps)) / np.ptp(inps)  # Scale [0, 1]

    for e in range(cfg["Epochs"]):  # Move e batches through network (80)

        # Get random batch
        ridx = np.random.randint(42, size=cfg["batch_size"])
        b_inps = inps[ridx]
        b_tars = tars[ridx]

        for d in range(cfg["batch_size"]):  # Move d examples through network (32)
            # Get X and T
            M = ut.initialize_log()  # Except weights

            M['X'] = b_inps[d]
            M['T'] = b_tars[d]
            M['X'] = np.repeat(M['X'], cfg["Steps"], axis=0)
            M['T'] = np.repeat(M['T'], cfg["Steps"], axis=0)


            M["XZ"] = rng.binomial(n=1, p=M["X"])


            for t in range(cfg["Steps"]):

                # Input is nonzero for first layer
                for r in range(cfg['N_Rec']):
                    # Spike if V >= threshold
                    M['Z'][t, r] = ut.eprop_Z(t=t,
                                              TZ=M['TZ'][r],
                                              V=M['V'][t, r],
                                              U=M['U'][t, r])
                    if t == cfg["Steps"] - 1:
                        break

                    # Pad any input with zeros to make it length N_R
                    Z_prev = M['Z'][t, r-1] if r > 0 else np.pad(
                        M['XZ'][t], (0, cfg["N_R"]-len(M['XZ'][t+1])))

                    M["Z_in"][t, r] = np.concatenate((Z_prev, M['Z'][t, r]))

                    M['I'][t, r] = np.dot(W['W'][t, r], M["Z_in"][t, r])

                    # if t > 0:
                    #     M["Zbar"][t, r] = cfg["alpha"] * M["Zbar"][t-1, r] + M["Z_in"][t, r]

                    M['H'][t, r] = ut.eprop_H(t=t,
                                              V=M['V'][t, r],
                                              U=M['U'][t, r],
                                              is_ALIF=M['is_ALIF'][r])

                    # Zt_in_prev = ut.temporal_filter(c=cfg["alpha"], a=M["Z_in"][:t+1, r])

                    M['EVV'][t+1, r] = ut.eprop_EVV(EVV=M['EVV'][t, r],
                                                    Z_in=M["Z_in"][t, r])

                    # TODO: Can do without M[ET] or M[H] or M[TZ] or M[DW].
                    M['EVU'][t+1, r] = ut.eprop_EVU(EVV=M['EVV'][t, r],
                                                    EVU=M['EVU'][t, r],
                                                    H=M['H'][t, r])
                    # print(M['EVV'][t, r].shape)
                    M['ET'][t, r] = ut.eprop_ET(H=M['H'][t, r],
                                                EVV=M['EVV'][t, r],
                                                EVU=M['EVU'][t, r],
                                                is_ALIF=M['is_ALIF'][r])
                    if t > 0:
                        M['ETbar'][t, r] = cfg["kappa"] * M["ETbar"][t-1, r] + M["ET"][t, r]

                    M['V'][t+1, r] = ut.eprop_V(V=M['V'][t, r],
                                                U=M['U'][t, r],
                                                I=M['I'][t, r],
                                                Z=M['Z'][t, r])

                    M['U'][t+1, r] = ut.eprop_U(U=M['U'][t, r],
                                                V=M['V'][t, r],
                                                Z=M['Z'][t, r],
                                                is_ALIF=M['is_ALIF'][r])  # Aggregate weights only

                    if t != cfg["Steps"] - 1:

                        # Calculate network output
                        M['Y'][t] = (cfg["kappa"] * M['Y'][t-1]
                                     + np.sum(W['W_out'][e] * M['Z'][t, -1])
                                     + W['b_out'][e])

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

                        M['DW'][t] = M['DW'][t-1] - cfg["eta"] * np.sum(
                            M['B'] * M["error"][t]) * ut.temporal_filter(cfg["kappa"], M['ET'][:t+1])

                        # freeze input weights
                        M['DW'][t, 0, :, :cfg["N_R"]].fill(0.)

                        M["DW_out"][t] = -cfg["eta"] * np.sum(
                            M["error"][t]) * ut.temporal_filter(cfg["kappa"], M['Z'][:t+1, -1])
                        M["Db_out"][t] = -cfg["eta"] * np.sum(W["error"][t])

                if (t > 0
                        and cfg["plot_interval"]
                        and ((t+1) % cfg["plot_interval"] == 0)):
                    vis.plot_drsnn(M=M,
                                   t=t,
                                   layers=None,   # Tuple for 2, None for heatmap
                                   neurons=None)  # Idem  # Move example through network for t steps (5 ms)

        # Update weights

        if cfg["update_dead_weights"]:
            for r2 in range(cfg["N_Rec"]):
                # Zero diag E: no self-conn
                for t in cfg["Steps"]:
                    np.fill_diagonal(M['DW'][t, r2, :, cfg["N_R"]:], 0)
        else:
            # Don't update zero-weights
            M['DW'] = np.where(W['W'], M['DW'], 0.)

        W["B"][e+1] = W["B"][e] + np.sum(M["DW_out"])
        W["B"][e+1] *= cfg["weight_decay"]

        W['W'][e+1] = W['W'][e] + np.sum(M['DW'])
        W["W_out"][e+1] = W['W_out'][t] + np.sum(M['DW_out'])
        W["W_out"][e+1] *= cfg["weight_decay"]

        W["b_out"][e+1] = W["b_out"][e] * np.sum(M["Db_out"])

        # Test on validation set

    # Return lowest validation error

    return np.mean(M["loss"])


if __name__ == "__main__":

    print(run_rsnn(cfg=CFG))