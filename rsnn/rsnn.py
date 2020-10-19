import matplotlib.pyplot as plt
import numpy as np
from config import cfg
import utils as ut
from task import task1


def run_rsnn(cfg):
    plot_interval = 1

    # Variable arrays
    # N = ut.initialize_neurons()
    # W = ut.initialize_weights()
    M = ut.initialize_log()

    Mt = {}

    if plot_interval != 0:
        fig = plt.figure(constrained_layout=False)
        gsc = fig.add_gridspec(nrows=max(8, 2 * cfg["N_Rec"] - 1),
                               ncols=4,
                               hspace=0.2)

        plt.ion()

    for ep in range(0, cfg["Epochs"]):

        # `ep+1' idx is future, operating on ep.

        # input is nonzero for first layer
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
                    t=ep,
                    uses_weights=True,
                    r=r)

            for key, item in Mt.items():
                if key in ["V", "U", "Z", "TZ", "H"]:
                    M[key][ep+1, r, :] = item[:cfg["N_R"]]
                elif key in ["W", "ET", "EVV", "EVU"]:
                    M[key][ep+1, r, :, :] = item

        # ERROR AND OUTPUT COLLECTION ##################################

        M["output"][ep, :] = M['Z'][ep, -1, :cfg["N_O"]]
        M["target"][ep, :] = task1(io_type="O", t=ep)

        # Exponential moving average
        for arrname in ["output", "target"]:
            M[f"{arrname}_EMA"][ep, :] = (
                cfg["EMA"] * M[arrname][ep, :]
                + (1 - cfg["EMA"]) * M[f"{arrname}_EMA"][ep, :]) if ep \
                else M[arrname][ep, :]

        error = np.mean(ut.errfn(M["output_EMA"][:ep+1, :],
                                 M["target_EMA"][:ep+1, :]),
                        axis=0)

        # Broadcast error to neurons next epoch
        M['L'][ep+1] = error * M['B'][ep]

        if plot_interval and (ep % plot_interval == 0):
            fig, gsc = ut.plot_drsnn(fig=fig,
                                     gsc=gsc,
                                     M=M,
                                     ep=ep,
                                     layers=(0, 1),
                                     neurons=(0, 0))

    return np.mean(ut.errfn(M["output_EMA"][:ep+1, :],
                            M["target_EMA"][:ep+1, :]),
                   axis=0)


if __name__ == "__main__":

    print(run_rsnn(cfg))

# TODO: Use ep-1, ep, ep+1?
# TODO: Why are the thresholds different in the rsnn and the units?
#       RSNN seems to spike at negative 65, but not units? Same for LIF?
# TODO: Why so many neurons in rsnn plot?
# TODO: Implement adaptive e-prop
# TODO: Dictionary to facilitate sweeping function (param = key, list = item)
# TODO: Merge drsnn plot and plot_logs
# TODO: Implement TIMIT
# TODO: Try replicate Bellec's results

"""
MEETING NOTES 10/13

* don't do too much, but do it right
  zwaartekracht NWO


* homeostaticity, slower dynamics in deeper layers. Neural sampling.
* Fading input not necessarily bad: think about it
* Learn nonlinear filter?
* nonlinear autoregressive moving average NARMA
* Input: drop autoregressive
* Search for NARMA-10 benchmark
* SEND SKYPE NAME




"""
