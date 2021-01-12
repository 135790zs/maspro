import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from matplotlib import rcParams as rc
import numpy as np
from graphviz import Digraph
from config import lookup

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


def weights_to_img(arr, is_binary=False):

    original_dim = arr.ndim
    arr = arr.reshape(arr.shape[0], -1).T

    # Pad to prevent spike trains disappearing on plot edge
    if is_binary:
        padsize = arr.shape[0]//10
        arr = np.pad(arr, ((padsize, padsize), (0, 0)))

    if original_dim == 4:
        # idx = np.argwhere(np.all(arr[..., :] == 0, axis=0))
        # arr_del = np.delete(arr, idx, axis=1)
        del_arr = arr[~np.all(arr == 0, axis=1)]
        if del_arr.shape[0] == 0:  # All synapses are zero, just pick the first
            arr = arr[:1]
        else:
            arr = del_arr
        # if del_arr.shape[0]:  # High enough to plot.

    # Don't plot too many weights. If too many, crop
    arr = arr[:(1000000//arr.shape[1])]

    return arr


def plot_run(cfg, R, W, epoch, log_id, inp_size):

    labelpad = 35
    fontsize = 14
    fig = plt.figure(constrained_layout=False, figsize=(8, 16))
    gsc = fig.add_gridspec(nrows=17 if cfg["Track_weights"] else 7,
                           ncols=1, hspace=0.05)
    axs = []

    for k, v in R.items():
        if k == 'eta':
            axs.append(fig.add_subplot(gsc[len(axs), :]))
            axs[-1].plot(v[:epoch])
        elif k == 'Hz':
            axs.append(fig.add_subplot(gsc[len(axs), :]))
            v = v[:epoch+1]
            v = v.reshape(epoch+1, -1) * 1000
            vm = np.mean(v[:epoch], axis=1)
            axs[-1].plot(vm)
            axs[-1].fill_between(np.arange(epoch),
                                 np.min(v[:epoch], axis=1),
                                 np.max(v[:epoch], axis=1),
                                 alpha=.25)
            # axs[-1].set_ylim(0, 100)
        elif type(v) == dict:
            axs.append(fig.add_subplot(gsc[len(axs), :]))
            for tvtype, arr in v.items():
                axs[-1].plot(arr[arr >= 0][:epoch], label=tvtype)
            axs[-1].legend()
        else:
            continue

        if k in ['err', f'%wrong']:
            axs[-1].set_yscale('log')
        axs[-1].grid()
        axs[-1].set_ylabel(k,
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)

    if epoch >= 1 and cfg["Track_weights"]:
        for weight_type, weights in W.items():
            axs.append(fig.add_subplot(gsc[len(axs), :]))
            if weight_type != 'W':

                # W_out, B, b_out
                if weight_type == "B":
                    weights = np.mean(weights, axis=3)
                elif weight_type == "W_out":
                    weights = np.mean(weights, axis=2)
                axs[-1].imshow(
                    weights_to_img(weights[:epoch]),
                    aspect='auto',
                    interpolation='nearest',
                    cmap='coolwarm')
                axs[-1].set_ylabel(f"{weight_type}"
                                   f"\n{np.min(weights[:epoch]):.1e}"
                                   f"\n{np.max(weights[:epoch]):.1e}",
                                   rotation=0,
                                   labelpad=labelpad,
                                   fontsize=fontsize)
                axs.append(fig.add_subplot(gsc[len(axs), :]))
                axs[-1].imshow(
                    weights_to_img(weights[1:epoch+1] - weights[:epoch]),
                    aspect='auto',
                    interpolation='nearest',
                    cmap='coolwarm')
                axs[-1].set_ylabel(f"D{weight_type}"
                                   f"\n{np.min(weights[1:epoch+1] - weights[:epoch]):.1e}"
                                   f"\n{np.max(weights[1:epoch+1] - weights[:epoch]):.1e}",
                                   rotation=0,
                                   labelpad=labelpad,
                                   fontsize=fontsize)
            else:
                # W_in
                a = weights[:epoch, :, 0, :, :cfg["N_R"]]
                axs[-1].imshow(
                    weights_to_img(np.mean(a, axis=3)),
                    aspect='auto',
                    interpolation='nearest',
                    cmap='coolwarm')
                axs[-1].set_ylabel(f"W_in"
                                   f"\n{np.min(a):.1e}"
                                   f"\n{np.max(a):.1e}",
                                   rotation=0,
                                   labelpad=labelpad,
                                   fontsize=fontsize)

                axs.append(fig.add_subplot(gsc[len(axs), :]))
                axs[-1].imshow(
                    weights_to_img(np.mean(weights[1:epoch+1, :, 0, :, :cfg["N_R"]], axis=3) -
                                   np.mean(weights[:epoch, :, 0, :, :cfg["N_R"]], axis=3)),
                    aspect='auto',
                    interpolation='nearest',
                    cmap='coolwarm')
                axs[-1].set_ylabel(f"DW_in"
                                   f"\n{np.min(weights[1:epoch+1, :, 0, :, :cfg['N_R']] - weights[:epoch, :, 0, :, :cfg['N_R']]):.1e}"
                                   f"\n{np.max(weights[1:epoch+1, :, 0, :, :cfg['N_R']] - weights[:epoch, :, 0, :, :cfg['N_R']]):.1e}",
                                   rotation=0,
                                   labelpad=labelpad,
                                   fontsize=fontsize)

                # W_rec
                axs.append(fig.add_subplot(gsc[len(axs), :]))
                W_rec0 = np.mean(weights[:epoch, :, 0, :, cfg["N_R"]:], axis=3).reshape(epoch, -1)
                if cfg["N_Rec"] > 1:
                    W_rec0 = np.concatenate((W_rec0, weights[:epoch, :, 1:].reshape(epoch, -1)), axis=1)
                W_rec1 = np.mean(weights[1:epoch+1, :, 0, :, cfg["N_R"]:], axis=3).reshape(epoch, -1)
                if cfg["N_Rec"] > 1:
                    W_rec1 = np.concatenate((W_rec1, weights[1:epoch+1, :, 1:].reshape(epoch, -1)), axis=1)
                axs[-1].imshow(
                    weights_to_img(W_rec0),
                    aspect='auto',
                    interpolation='nearest',
                    cmap='coolwarm')
                axs[-1].set_ylabel(f"W_rec"
                                   f"\n{np.min(W_rec0):.1e}"
                                   f"\n{np.max(W_rec0):.1e}",
                                   rotation=0,
                                   labelpad=labelpad,
                                   fontsize=fontsize)

                axs.append(fig.add_subplot(gsc[len(axs), :]))
                a = W_rec1 - W_rec0
                axs[-1].imshow(
                    weights_to_img(a),
                    aspect='auto',
                    interpolation='nearest',
                    cmap='coolwarm')
                axs[-1].set_ylabel(f"DW_rec"
                                   f"\n{np.min(a):.1e}"
                                   f"\n{np.max(a):.1e}",
                                   rotation=0,
                                   labelpad=labelpad,
                                   fontsize=fontsize)

    axs[-1].set_xlabel("Epoch $E$", fontsize=fontsize)

    plt.savefig(f"../log/{log_id}/metric.pdf",
                bbox_inches='tight')
    plt.savefig(f"../vis/latest_metric.pdf",
                bbox_inches='tight')

    plt.close()


def plot_state(cfg, M, B, W_rec, W_out, b_out, e, it, log_id, plot_weights=False):
    S_plotvars = ["X", "I_in", "I_rec", "I", "V", "a", "A", "Z", "H"]
    if cfg["Track_synapse"]:
        S_plotvars += ["EVV", "EVU", "ET", "ETbar", "gW"]
    S_plotvars += ["L_std", "L_reg", "spikerate", "Y"]

    M_plotvars = ["P", "D", "Pmax", "T", "Correct", "CE"]
    if cfg["n_directions"] > 1:
        M_plotvars = ["X", "Y"] + M_plotvars  # Show combined/corrected in, out

    W = {"B": B, "W_rec": W_rec, "W_out": W_out, "b_out": b_out}

    nrows = (len(M_plotvars)
             + len(S_plotvars)
             + (len(W.keys()) if plot_weights else 0))

    fig = plt.figure(constrained_layout=False, figsize=(8, nrows//1.2))
    gsc = fig.add_gridspec(nrows=nrows,
                           ncols=cfg["n_directions"],
                           hspace=0.075,
                           wspace=0.5)
    axs = []
    labelpad = 30
    fontsize = 13

    fig.suptitle(f"Single-run model state\n"
                 f"ID {log_id}, Epoch {e}, Iter {it}",
                 fontsize=20)
    row_idx = 0

    for var in S_plotvars:
        for s in range(cfg["n_directions"]):
            axs.append(fig.add_subplot(gsc[row_idx, s]))
            if var == "X":
                arr = M[f"X{s}"]
            else:
                arr = M[var][s]
            # For synapses, plot mean of set with shared receiving.
            # if arr.ndim == 4:
            #     arr = np.mean(arr, axis=3)

            axs[-1].imshow(weights_to_img(arr,
                                          is_binary=lookup[var]["binary"]),
                           cmap=('copper' if lookup[var]["binary"]
                                 else 'coolwarm'),
                           vmin=np.min(arr),
                           vmax=np.max(arr),
                           interpolation="none",  # better than nearest here
                           aspect='auto')

            axs[-1].set_ylabel(f"${lookup[var]['label']}$"
                               f"\n{np.min(arr):.1e}"
                               f"\n{np.max(arr):.1e}",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        row_idx += 1

    for var in M_plotvars:

        axs.append(fig.add_subplot(gsc[row_idx, :]))
        row_idx += 1
        if var == 'Y':
            # TODO: make special var Yc for combined Y
            arr = M['Y'][0] + (np.flip(M['Y'][1], axis=0)
                               if cfg["n_directions"] > 1 else 0)
        elif var == 'X' and cfg["n_directions"] > 1:
            arr = M["X0"]
        else:
            arr = M[var]

        if not lookup[var]['scalar']:
            axs[-1].imshow(weights_to_img(arr,
                                          is_binary=lookup[var]["binary"]),
                           cmap=('RdYlGn' if var == 'Correct' else
                                 'copper' if lookup[var]["binary"]
                                 else 'coolwarm'),
                           vmin=np.min(arr),
                           vmax=np.max(arr),
                           interpolation="none",  # better than nearest here
                           aspect='auto')

        elif var == 'CE':  # not as image but line plot
            axs[-1].plot(arr)
            axs[-1].margins(0)
            axs[-1].grid()

        axs[-1].set_ylabel(f"${lookup[var]['label']}$"
                           f"\n{np.min(arr):.1e}"
                           f"\n{np.max(arr):.1e}",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)

    if plot_weights:
        for k, v in W.items():
            if k in ["W_rec", "B"]:
                v = np.mean(v, axis=3)
            if k == "W_out":
                v = np.mean(v, axis=2)
            v = v.flatten()
            v = np.expand_dims(v, axis=0)
            v = np.repeat(v, M['X0'].shape[0], axis=0)

            axs.append(fig.add_subplot(gsc[row_idx, :],
                                       sharex=axs[0] if axs else None))
            row_idx += 1
            axs[-1].imshow(weights_to_img(v),
                           cmap='coolwarm',
                           vmin=np.min(v),
                           vmax=np.max(v),
                           interpolation='nearest',
                           aspect='auto')
            axs[-1].set_ylabel(f"${lookup[k]['label']}$"
                               f"\n{np.min(v):.1e}"
                               f"\n{np.max(v):.1e}",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

    axs[-1].set_xlabel("$t$", fontsize=fontsize)

    plt.savefig(f"../log/{log_id}/states/state.pdf",
                bbox_inches='tight')
    plt.savefig(f"../vis/latest_state.pdf",
                bbox_inches='tight')
    if e % cfg["state_save_interval"] == 0:
        plt.savefig(f"../log/{log_id}/states/state_{e}.pdf",
                    bbox_inches='tight')

    plt.close()


def plot_pair(cfg, M, B, W_rec, W_out, b_out, e, log_id):
    S_plotvars = ["I", "V", "a", "Z", "H"]
    if cfg["Track_synapse"]:
        S_plotvars += ["EVV", "EVU", "ET", "ETbar"]
    S_plotvars += ["L_std"]
    if cfg["Track_synapse"]:
        S_plotvars += ["gW", "DW"]

    fig = plt.figure(constrained_layout=False, figsize=(8, len(S_plotvars)//1.2))
    gsc = fig.add_gridspec(nrows=len(S_plotvars),
                           ncols=1,
                           hspace=0.075,
                           wspace=0.5)
    axs = []
    labelpad = 30
    fontsize = 13

    rng = np.random.default_rng(5)
    ni = 0
    nj = 0
    tries = 0
    while True:
        nj = rng.integers(cfg["N_R"])
        nj = rng.integers(cfg["N_R"])
        if (nj != ni
            and W_rec[0, 0, nj, ni+cfg["N_R"]] != 0
            and (tries >= 100  # Prevents infloop if no spikes at all.
                 or (M['Z'][0, :, 0, ni].any()
                     and M['Z'][0, :, 0, nj].any()))):
            break
        tries += 1

    row_idx = 0
    for var in S_plotvars:
        axs.append(fig.add_subplot(gsc[row_idx]))
        if var == "DW":
            arr = np.cumsum(M['gW'][0, :, 0], axis=0)
        else:
            arr = M[var][0, :, 0]

        if arr.ndim == 2:
            axs[-1].plot(arr[:, ni], linewidth=0.2, label=f"presynaptic: i={ni}")
            axs[-1].plot(arr[:, nj], linewidth=0.2, label=f"postsynaptic: j={nj}")

        elif arr.ndim == 3:
            axs[-1].plot(arr[:, nj, ni+cfg["N_R"]], linewidth=0.2)

        axs[-1].set_ylabel(f"${lookup[var]['label']}$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)
        # axs[-1].set_xticks(np.arange(0, arr.shape[0], 1))
        # axs[-1].grid(axis='x')
        row_idx += 1

    axs[-1].set_xlabel("$t$", fontsize=fontsize)
    axs[0].legend()

    plt.savefig(f"../vis/pair.pdf",
                bbox_inches='tight')

    plt.close()



def plot_graph(cfg, M, t, W_rec, W_out, log_id):
    dot = Digraph(format='svg', engine='dot')
    precision = 0.01

    # neurons
    def neuroncolor(r, n, spiked):
        if r is None:
            return None

        bounds = (0, cfg["thr"])
        v = (M['V'][t, r, n] - bounds[0]) / bounds[1]
        cmap = mpcm.get_cmap("coolwarm")
        rgba = cmap(v, bytes=True)
        ret = "#"
        for val in rgba:
            hval = hex(val)[2:]
            ret += hval if hval != '0' else '00'
        spike_col = '#33ff33' if spiked else ret
        return spike_col + ';0.1:' + ret

    # TODO: compress loops using itertools
    for r in range(0, cfg["N_Rec"]):
        for n in range(0, cfg["N_R"]):
            spiked = bool(M['Z'][t, r, n])
            dot.node(name=f"{r}-{n}",
                     label=f"{r}-{n}\n{M['V'][t, r, n]:.2f}",
                     style='radial',
                     fixedsize='false',
                     color="#ffffff",
                     fillcolor=neuroncolor(r=r, n=n, spiked=spiked))
    for n in range(0, M['X'].shape[-1]):
        dot.node(name=f"in-{n}",
                 label=f"in-{n}\n{M['X'][t, n]:.2f}",
                 style='radial',
                 fixedsize='false',
                 fillcolor=("#ffffff" if M['X'][t, n] == 0
                            else "#33ff33;0.1:#ffffff"))

    # weights
    def weightcolor(w):
        maxdev = max(abs(np.min(W_rec)),
                     abs(np.max(W_rec)))
        w = (w / maxdev) + (1 / 2)
        cmap = mpcm.get_cmap("bwr")
        rgba = cmap(w, bytes=True)
        ret = "#"
        for val in rgba:
            ret += hex(val)[2:]
        return ret

    # in-to-rec
    for tail in range(0, M['X'].shape[-1]):
        for head in range(0, cfg["N_R"]):
            rdw = np.sum(M['DW'][:, 0, head, tail])
            added = (('+' if rdw >= 0 else '-')
                     + f'{abs(rdw):.2f}')
            dot.edge(tail_name=f"in-{tail}",
                     head_name=f"{0}-{head}",
                     label=f"{W_rec[0, head, tail]:.2f}" +
                           (f"({added})" if abs(rdw) > precision else ''),
                     penwidth='1',
                     color=weightcolor(w=W_rec[0, head, tail]))

    # intra-rec
    for r in range(0, cfg["N_Rec"]):  # r to r+1
        for head in range(0, cfg["N_R"]):
            for tail in range(cfg["N_R"], 2*cfg["N_R"]):
                rdw = np.sum(M['DW'][:, r, head, tail])
                added = (('+' if rdw >= 0 else '-')
                         + f'{abs(rdw):.2f}')
                if W_rec[r, head, tail] != 0:
                    dot.edge(tail_name=f"{r}-{tail-cfg['N_R']}",
                             head_name=f"{r}-{head}",
                             label=f"{W_rec[r, head, tail]:.2f}" +
                                   (f"({added})" if abs(
                                    rdw) > precision
                                    else ''),
                             penwidth='1',
                             color=weightcolor(w=W_rec[r, head, tail]))

    # inter-rec
    for r in range(0, cfg["N_Rec"]-1):
        for head in range(0, cfg["N_R"]):
            for tail in range(0, cfg["N_R"]):
                if M['W'][t, r, head, tail] != 0:
                    dot.edge(tail_name=f"{r}-{head}",
                             head_name=f"{r+1}-{tail}",
                             label=f"{W_rec[r, head, tail]:.2f}",
                             penwidth='1',
                             color=weightcolor(w=W_rec[r, head, tail]))

    # rec-to-out
    for tail in range(0, cfg["N_R"]):
        for head in range(0, W_out.shape[-2]):
            dot.edge(tail_name=f"{cfg['N_Rec']-1}-{tail}",
                     head_name=f"out-{head}",
                     label=f"{W_out[head, tail]:.2f}",
                     penwidth='1',
                     color=weightcolor(w=W_out[head, tail]))

    dot.attr(label=f"Steps {t+1}")
    dot.render(f"../log/{log_id}/net")
