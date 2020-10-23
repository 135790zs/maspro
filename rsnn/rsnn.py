import numpy as np
from config import cfg as CFG
import utils as ut
import vis
from task import task1, narma10


def run_rsnn(cfg=None):
    plot_interval = 1

    # Variable arrays
    M = ut.initialize_log()

    Mt = {}

    for ep in range(0, cfg["Epochs"]-1):

        # input is nonzero for first layer
        if cfg["task"] == "narma10":
            # input given in array
            pass
        else:
            M["input"][ep, :] = task1(io_type="I", t=ep)

        # Bernoulli distribtion
        rng = np.random.default_rng()
        M["input_spike"][ep, :] = rng.binomial(n=1, p=M["input"][ep, :])

        # Feed to input layer R0: let first layer exceed threshold if input
        M['V'][ep, 0, :cfg["N_I"]] = np.where(
            M["input_spike"][ep, :],
            (cfg["thr"] if cfg["neuron"] != "ALIF"
             else M['U'][ep, 0, :cfg["N_I"]]),
            M['V'][ep, 0, :cfg["N_I"]])

        for r in range(0, cfg["N_Rec"] - 1):
            for key, item in M.items():
                if key in ["V", "U", "Z", "TZ", "H"]:
                    Mt[key] = np.concatenate((item[ep, r], item[ep, r+1]))
                elif key == 'L':
                    Mt[key] = item[ep, r-1, :] if r > 0 else np.zeros(
                        shape=cfg["N_R"])
                elif key in ["W", "ET", "EVV", "EVU"]:
                    Mt[key] = item[ep, r, :, :]

            Mt = ut.eprop(
                model=cfg["neuron"],
                M=Mt,
                X=np.pad(array=M["input_spike"][ep, :],
                         pad_width=(0, 2 * cfg["N_R"] - cfg["N_I"])),
                t=ep)

            for key, item in Mt.items():
                if key in ["V", "U", "Z", "TZ", "H"]:
                    M[key][ep, r, :] = item[:cfg["N_R"]]
                    M[key][ep, r+1, :] = item[cfg["N_R"]:]
                elif key in ["W", "ET", "EVV", "EVU"]:
                    M[key][ep, r, :, :] = item

        for key, item in M.items():
            if key in ["V", "U", "Z", "TZ", "H", "W", "ET", "EVV", "EVU"]:
                M[key][ep+1] = item[ep]

        # ERROR AND OUTPUT COLLECTION ##################################
        M["output"][ep] = M['Z'][ep, -1, :cfg["N_O"]]
        M["output_EMA"][ep] = ut.EMA(arr=M["output"],
                                     arr_ema=M["output_EMA"],
                                     ep=ep)

        if cfg["task"] == "narma10":
            M["target"][ep] = narma10(t=ep,
                                      u=M["input"][:ep+1],
                                      y=M["output_EMA"][:ep+1])
        else:
            M["target"][ep] = task1(io_type="O", t=ep)

        M["target_EMA"][ep] = ut.EMA(arr=M["target"],
                                     arr_ema=M["target_EMA"],
                                     ep=ep)

        M["error"][ep] = np.sum(M["output"][ep] - M["target"][ep]) ** 2
        M["error_EMA"][ep] = ut.EMA(arr=M["error"],
                                    arr_ema=M["error_EMA"],
                                    ep=ep)

        # Broadcast error to neurons next epoch
        M['L'][ep+1] = M["error"][ep] * M['B'][ep]

        if plot_interval and (ep % plot_interval == 0):
            vis.plot_drsnn(M=M,
                           ep=ep,
                           layers=(1, 2),
                           neurons=(0, 0))

    return np.mean(M["error"])


if __name__ == "__main__":

    print(run_rsnn(CFG))


#
"""
FOR THE NEXT TIME:

Rewrite the whole system. Just use Bellec's model, first single-layer and then stacked. Then apply Traub's LIF fix. Then Izhikevich.

"""
