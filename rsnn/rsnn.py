import matplotlib.pyplot as plt
import numpy as np
from config import cfg
import utils as ut
from task import task1


def run_rsnn(cfg):
    plot_interval = 1

    # Variable arrays
    N = ut.initialize_neurons()
    W = ut.initialize_weights()
    log = ut.initialize_log()

    if plot_interval != 0:
        fig = plt.figure(constrained_layout=False)
        gsc = fig.add_gridspec(nrows=max(8, 2 * cfg["N_Rec"] - 1),
                               ncols=4,
                               hspace=0.2)

        plt.ion()

    for ep in range(0, cfg["Epochs"]):

        # input is nonzero for first layer
        log["input"][ep, :] = task1(io_type="I", t=ep)
        # Bernoulli distribtion
        rng = np.random.default_rng()
        log["input_spike"][ep, :] = rng.binomial(n=1, p=log["input"][ep, :])

        # Feed to input layer R0
        N['V'][0, :cfg["N_I"]] = np.where(
            log["input_spike"][ep, :],
            cfg["thr"] if cfg["neuron"] != "ALIF"
            else N['U'][0, :cfg["N_I"]],
            N['V'][0, :cfg["N_I"]])
        for r in range(0, cfg["N_Rec"] - 1):
            print(N['V'])
            N_concat, W_layer = ut.eprop(
                    model=cfg["neuron"],
                    V=np.concatenate((N['V'][r, :], N['V'][r+1, :])),
                    U=np.concatenate((N['U'][r, :], N['U'][r+1, :])),
                    Z=np.concatenate((N['Z'][r, :], N['Z'][r+1, :])),
                    TZ=np.concatenate((N['TZ'][r, :], N['TZ'][r+1, :])),
                    EVV=W['EVV'][r, :, :],
                    EVU=W['EVU'][r, :, :],
                    W=W['W'][r, :, :],
                    L=W['L'][r-1, :] if r > 0 else np.zeros(shape=cfg["N_R"]),
                    X=np.pad(array=log["input_spike"][ep, :],
                             pad_width=(0, 2*cfg["N_R"]-cfg["N_I"])),
                    t=ep)

            for key, item in N_concat.items():
                N[key][r, :] = item[:cfg["N_R"]]
                if key != "TZ":
                    log[key][ep, :, :] = N[key]

            for key, item in W_layer.items():
                W[key][r, :, :] = item
                log[key][ep, :, :, :] = W[key]


            # N['V'][r, :] = Nvr[:cfg["N_R"]]
            # if cfg["neuron"] not in ["LIF"]:
            #     N['U'][r, :] = Nur[:cfg["N_R"]]
            # N['Z'][r, :] = Nzr[:cfg["N_R"]]
            # N['TZ'][r, :] = TZr[:cfg["N_R"]]
            # N['H'][r, :] = Hr[:cfg["N_R"]]

            # log["V"][ep, :, :] = N['V']
            # log["U"][ep, :, :] = N['U']
            # log["Z"][ep, :, :] = N['Z']
            # log["H"][ep, :, :] = N['H']
            # log["ET"][ep, :, :, :] = W['ET']
            # log["EVV"][ep, :, :, :] = W['EVV']
            # log["EVU"][ep, :, :, :] = W['EVU']
            # log["W"][ep, :, :, :] = W['W']

        # ERROR AND OUTPUT COLLECTION ##################################

        log["output"][ep, :] = N['Z'][-1, :cfg["N_O"]]
        log["target"][ep, :] = task1(io_type="O", t=ep)

        # Exponential moving average
        for arrname in ["output", "target"]:
            log[f"{arrname}_EMA"][ep, :] = (
                cfg["EMA"] * log[arrname][ep, :]
                + (1 - cfg["EMA"]) * log[f"{arrname}_EMA"][ep-1, :]) if ep \
                else log[arrname][ep, :]

        error = np.mean(ut.errfn(log["output_EMA"][:ep+1, :],
                                 log["target_EMA"][:ep+1, :]),
                        axis=0)

        # Broadcast error to neurons next epoch
        W['L'] = error * W['B']

        if plot_interval and (ep % plot_interval == 0):
            fig, gsc = ut.plot_drsnn(fig=fig,
                                     gsc=gsc,
                                     V=N['V'],
                                     W=W['W'],
                                     Z=N['Z'],
                                     log=log,
                                     ep=ep,
                                     layers=(0, 1),
                                     neurons=(0, 0))

    return np.mean(ut.errfn(log["output_EMA"][:ep+1, :],
                            log["target_EMA"][:ep+1, :]),
                   axis=0)


if __name__ == "__main__":

    print(run_rsnn(cfg))

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
