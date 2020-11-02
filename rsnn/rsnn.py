import numpy as np
from config import cfg as CFG
import utils as ut
import vis
from task import narma10


def run_rsnn(cfg):

    # Variable arrays
    M = ut.initialize_log()

    for t in range(cfg["Epochs"]):

        # Input is nonzero for first layer
        for r in range(cfg['N_Rec']):
            # Spike if V >= threshold
            M['Z'][t, r] = ut.eprop_Z(t=t,
                                      TZ=M['TZ'][r],
                                      V=M['V'][t, r],
                                      U=M['U'][t, r])
            if t == cfg["Epochs"] - 1:
                break

            # Pad any input with zeros to make it length N_R
            Z_prev = M['Z'][t, r-1] if r > 0 else np.pad(
                M['XZ'][t], (0, cfg["N_R"]-len(M['XZ'][t+1])))

            M["Z_in"][t, r] = np.concatenate((Z_prev, M['Z'][t, r]))

            M['I'][t, r] = np.dot(M['W'][t, r], M["Z_in"][t, r])

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
                                        is_ALIF=M['is_ALIF'][r])

        if t != cfg["Epochs"] - 1:

            # Calculate network output
            M['Y'][t] = (cfg["kappa"] * M['Y'][t-1]
                         + np.sum(M['W_out'] * M['Z'][t, -1])
                         + M['b_out'][t])

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
                print(M["error"][t])
                M["loss"][t] = -np.sum(M['T'][t] * np.log(sm))

            M['DW'][t] = -cfg["eta"] * np.sum(
                M['B'] * M["error"][t]) * ut.temporal_filter(cfg["kappa"], M['ET'][:t+1])

            # freeze input weights
            M['DW'][t, 0, :, :cfg["N_R"]].fill(0.)

            if cfg["update_dead_weights"]:
                for r2 in range(cfg["N_Rec"]):
                    # Zero diag E: no self-conn
                    np.fill_diagonal(M['DW'][t, r2, :, cfg["N_R"]:], 0)
            else:
                # Don't update zero-weights
                M['DW'][t] = np.where(M['W'][t], M['DW'][t], 0.)


            M["DW_out"][t] = -cfg["eta"] * np.sum(
                M["error"][t]) * ut.temporal_filter(cfg["kappa"], M['Z'][:t+1, -1])

            M["B"][t+1] = M["B"][t] + M["DW_out"][t]
            M["B"][t+1] *= cfg["weight_decay"]

            M['W'][t+1] = M['W'][t] + M['DW'][t]

            M["W_out"][t+1] = M['W_out'][t] + M['DW_out'][t]
            M["W_out"][t+1] *= cfg["weight_decay"]

            M["b_out"][t+1] = M["b_out"][t] - cfg["eta"] * np.sum(M["error"][t])

        if (t > 0
                and cfg["plot_interval"]
                and ((t+1) % cfg["plot_interval"] == 0)):
            vis.plot_drsnn(M=M,
                           t=t,
                           layers=None,   # Tuple for 2, None for heatmap
                           neurons=None)  # Idem

    return np.mean(M["loss"])


if __name__ == "__main__":

    print(run_rsnn(cfg=CFG))

# RM Traub's contribs
