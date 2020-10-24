import numpy as np
from config import cfg as CFG
import utils as ut
import vis


def run_rsnn(cfg=None, **kwargs):

    # Variable arrays
    M = ut.initialize_log()  # V, Z, EVV, X, Y

    for t in range(cfg["Epochs"]):

        # input is nonzero for first layer
        for r in range(cfg['N_Rec']):
            # Spike if V >= threshold
            M['Z'][t, r] = ut.eprop_Z(t=t,
                                      TZ=M['TZ'][r],
                                      V=M['V'][t, r],
                                      U=M['U'][t, r])

            # Log time of spike
            M['TZ'][r] = np.where(M['Z'][t, r], t, M['TZ'][r])

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
                                            Z=M['Z'][t, r])

            # TODO: Can do without M[ET] or M[H] or M[TZ] or M[DW].
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

            M['DW'][t] = -cfg["eta"] * np.sum(
                M['B'] * M["error"][t+1]) * M['ET'][t+1]

            # Don't update dropped weights
            M['DW'][t] = np.where(M['W'][t], M['DW'][t], 0.)

            M['W'][t+1] = M['W'][t] + M['DW'][t]

        if (t > 0
                and cfg["plot_interval"]
                and ((t+1) % cfg["plot_interval"] == 0)):
            vis.plot_drsnn(M=M,
                           t=t,
                           fname=kwargs['fname'],
                           layers=kwargs['layers'],
                           neurons=kwargs['neurons'])

    return np.mean(M["error"])


if __name__ == "__main__":

    print(run_rsnn(CFG, layers=(0, 1), neurons=(0, 0), fname=""))
