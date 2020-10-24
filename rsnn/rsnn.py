import numpy as np
from config import cfg as CFG
import utils as ut
import vis
from task import narma10
import time


def run_rsnn(cfg=None, **kwargs):

    # Variable arrays
    M = ut.initialize_log()  # V, Z, EVV, X, Y

    for t in range(cfg["Epochs"]):

        print(f"\n{'#'*10} TIME = {t} {'#'*10}\n")
        # input is nonzero for first layer
        for r in range(cfg['N_Rec']):
            # Spike if V >= threshold
            M['Z'][t, r] = ut.eprop_Z(t=t,
                                      TZ=M['TZ'][r],
                                      V=M['V'][t, r],
                                      U=M['U'][t, r])

            # Log time of spike
            M['TZ'][r] = np.where(M['Z'][t, r], t, M['TZ'][r])  # TODO: Remove TZ

            # Pad any input with zeros to make it length N_R
            Z_in = np.concatenate((M['Z'][t, r-1] if r > 0 else
                                   np.pad(np.asarray([M['XZ'][t]]),
                                          (0, cfg["N_R"]-1)),
                                   M['Z'][t, r]))

            I = np.dot(M['W'][t, r], Z_in)

            R = (t - M['TZ'][r] <= cfg["dt_refr"])
            if t == cfg["Epochs"] - 1:
                break

            M['V'][t+1, r] = ut.eprop_V(V=M['V'][t, r],
                                        U=M['U'][t, r],
                                        I=I,
                                        Z=M['Z'][t, r],
                                        R=R)

            M['U'][t+1, r] = ut.eprop_U(U=M['U'][t, r],
                                        V=M['V'][t, r],
                                        Z=M['Z'][t, r])

            M['EVV'][t+1, r] = ut.eprop_EVV(EVV=M['EVV'][t, r],
                                            EVU=M['EVU'][t, r],
                                            V=M['V'][t, r],
                                            Z=M['Z'][t, r],
                                            R=R,
                                            Z_in=Z_in)

            M['H'][t+1, r] = ut.eprop_H(t=t,
                                        TZ=M['TZ'][r],
                                        V=M['V'][t+1, r],
                                        U=M['U'][t+1, r])

            M['EVU'][t+1, r] = ut.eprop_EVU(EVV=M['EVV'][t, r],
                                            EVU=M['EVU'][t, r],
                                            H=M['H'][t, r],  # ALIF
                                            Z=M['Z'][t, r],
                                            R=R)

            # Can do without M[ET] or M[H] or M[TZ] or M[DW].
            M['ET'][t+1, r] = ut.eprop_ET(H=M['H'][t+1, r],
                                          EVV=M['EVV'][t+1, r],
                                          EVU=M['EVU'][t+1, r])

        if t != cfg["Epochs"] - 1:

            # Calculate network output
            M['Y'][t+1] = (cfg["kappa"] * M['Y'][t]
                           + np.sum(M['W_out'] * M['Z'][t+1, -1])
                           + M['b_out'])

            # For some tasks, the desired output is the source of the input
            M["error"][t+1] = (M["Y"][t+1] - M["T"][t+1]) ** 2


            M['DW'][t] = -cfg["eta"] * np.sum(M['B'] * M["error"][t+1]) * M['ET'][t+1]
            M['DW'][t] = np.where(M['W'][t], M['DW'][t], 0.)  # Don't update dropped weights

            M['W'][t+1] = M['W'][t] + M['DW'][t]

        if cfg["plot_interval"] and ((t+1) % cfg["plot_interval"] == 0):
            vis.plot_drsnn(M=M,
                           t=t,
                           layers=kwargs['layers'],
                           neurons=kwargs['neurons'],
                           fname=kwargs['fname'])

    return np.mean(M["error"])


# def run_rsnn2(cfg=None):
#     plot_interval = 1

#     # Variable arrays
#     M = ut.initialize_log()

#     Mt = {}

#     for ep in range(0, cfg["Epochs"]-1):

#         # input is nonzero for first layer
#         if cfg["task"] == "narma10":
#             # input given in array
#             pass
#         else:
#             M["input"][ep, :] = task1(io_type="I", t=ep)

#         # Bernoulli distribtion
#         rng = np.random.default_rng()
#         M["input_spike"][ep, :] = rng.binomial(n=1, p=M["input"][ep, :])

#         # Feed to input layer R0: let first layer exceed threshold if input
#         M['V'][ep, 0, :cfg["N_I"]] = np.where(
#             M["input_spike"][ep, :],
#             (cfg["thr"] if cfg["neuron"] != "ALIF"
#              else M['U'][ep, 0, :cfg["N_I"]]),
#             M['V'][ep, 0, :cfg["N_I"]])

#         for r in range(0, cfg["N_Rec"] - 1):
#             for key, item in M.items():
#                 if key in ["V", "U", "Z", "TZ", "H"]:
#                     Mt[key] = np.concatenate((item[ep, r], item[ep, r+1]))
#                 elif key == 'L':
#                     Mt[key] = item[ep, r-1, :] if r > 0 else np.zeros(
#                         shape=cfg["N_R"])
#                 elif key in ["W", "ET", "EVV", "EVU"]:
#                     Mt[key] = item[ep, r, :, :]

#             Mt = ut.eprop(
#                 model=cfg["neuron"],
#                 M=Mt,
#                 X=np.pad(array=M["input_spike"][ep, :],
#                          pad_width=(0, 2 * cfg["N_R"] - cfg["N_I"])),
#                 t=ep)

#             for key, item in Mt.items():
#                 if key in ["V", "U", "Z", "TZ", "H"]:
#                     M[key][ep, r, :] = item[:cfg["N_R"]]
#                     M[key][ep, r+1, :] = item[cfg["N_R"]:]
#                 elif key in ["W", "ET", "EVV", "EVU"]:
#                     M[key][ep, r, :, :] = item

#         for key, item in M.items():
#             if key in ["V", "U", "Z", "TZ", "H", "W", "ET", "EVV", "EVU"]:
#                 M[key][ep+1] = item[ep]

#         # ERROR AND OUTPUT COLLECTION ##################################
#         M["output"][ep] = M['Z'][ep, -1, :cfg["N_O"]]
#         M["output_EMA"][ep] = ut.EMA(arr=M["output"],
#                                      arr_ema=M["output_EMA"],
#                                      ep=ep)

#         if cfg["task"] == "narma10":
#             M["target"][ep] = narma10(t=ep,
#                                       u=M["input"][:ep+1],
#                                       y=M["output_EMA"][:ep+1])
#         else:
#             M["target"][ep] = task1(io_type="O", t=ep)

#         M["target_EMA"][ep] = ut.EMA(arr=M["target"],
#                                      arr_ema=M["target_EMA"],
#                                      ep=ep)

#         M["error"][ep] = np.sum(M["output"][ep] - M["target"][ep]) ** 2
#         M["error_EMA"][ep] = ut.EMA(arr=M["error"],
#                                     arr_ema=M["error_EMA"],
#                                     ep=ep)

#         # Broadcast error to neurons next epoch
#         M['L'][ep+1] = M["error"][ep] * M['B'][ep]

#         if plot_interval and (ep % plot_interval == 0):
#             vis.plot_drsnn(M=M,
#                            ep=ep,
#                            layers=(1, 2),
#                            neurons=(0, 0))

#     return np.mean(M["error"])


if __name__ == "__main__":

    print(run_rsnn(CFG, layers=(0, 1), neurons=(0, 0), fname=""))


#
"""
FOR THE NEXT TIME:

Rewrite the whole system. Just use Bellec's model, first single-layer and then stacked. Then apply Traub's LIF fix. Then Izhikevich.

"""
