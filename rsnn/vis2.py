import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from matplotlib import rcParams as rc
import numpy as np
from config2 import lookup

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


def plot_M(cfg, M, it, log_id, n_steps, inp_size):
    plotvars = ['x', 'I_in', 'I_rec', 'I', "v", "a", "z", "h"]
    if cfg["Track_synapse"]:
        plotvars += ["vv", "va", "etbar"]
    plotvars += ['l_std', 'l_fr', 'l', "y", 'p', 'pm', 't', 'd', 'correct']

    fig = plt.figure(constrained_layout=False, figsize=(8, len(plotvars)//1.2))
    gsc = fig.add_gridspec(nrows=len(plotvars),
                           ncols=1,
                           hspace=0.075,
                           wspace=0.5)
    axs = []
    labelpad = 30
    fontsize = 13

    fig.suptitle(f"ID {log_id}, Iteration {it}", fontsize=20)
    row_idx = 0
    rng = np.random.default_rng(seed=cfg["seed"])
    b = rng.choice(n_steps.size)

    for var in plotvars:
        axs.append(fig.add_subplot(gsc[row_idx]))
        if var in ['y', 'p', 'pm', 't', 'd', 'correct']:
            arr = M[var][b, :n_steps[b]].cpu().numpy()
        else:
            arr = M[var][0, b, :n_steps[b]].cpu().numpy()
        if var == 'x':
            arr = arr[:, :inp_size]
        axs[-1].imshow(weights_to_img(arr,
                                      is_binary=lookup[var]["binary"]),
                       cmap=('RdYlGn' if var == 'correct' else
                                 'copper' if lookup[var]["binary"]
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


    axs[-1].set_xlabel("$t$", fontsize=fontsize)

    plt.savefig(f"../log/{log_id}/states/state.pdf",
                bbox_inches='tight')
    plt.savefig(f"../vis/latest_state.pdf",
                bbox_inches='tight')
    if it % cfg["state_save_interval"] == 0:
        plt.savefig(f"../log/{log_id}/states/it_{it}.pdf",
                    bbox_inches='tight')

    plt.close()


def interpolate_verrs(arr):
    """ Interpolate missing validation accuracy and loss values.

    These exist, because validation may not happen in all iterations.
    """

    # TODO: comments
    retarr = arr
    x0 = 0
    y0 = arr[x0]
    for idx, val in enumerate(arr):
        retarr[idx] = val

        if val != -1:
            x0 = idx
            y0 = val
            continue

        x1 = np.argmax(arr[idx:] > -1) + idx
        y1 = arr[x1]
        w = (idx - x0) / (x1 - x0)
        retarr[idx] = y0 * (1 - w) + y1 * w

    return retarr


def plot_W(cfg, W_log, log_id):
    labelpad = 35
    fontsize = 14
    fig = plt.figure(constrained_layout=False, figsize=(8, 16))
    gsc = fig.add_gridspec(nrows=11,
                           ncols=1, hspace=0.05)
    axs = []
    for k, v in W_log.items():
        if type(v) == dict:
            axs.append(fig.add_subplot(gsc[len(axs), :]))
            # v['val'] = interpolate_verrs(np.array(v['val']))
            for tvtype, arr in v.items():
                arr = interpolate_verrs(np.array(arr))
                axs[-1].plot(arr[arr >= 0], label=tvtype)
            axs[-1].legend()
            axs[-1].grid()
            if k in ['Cross-entropy', 'Percentage wrong']:
                axs[-1].set_yscale('log')
            axs[-1].set_ylabel(k,
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)
        if k in ['W', 'out', 'bias', 'B']:
            weights = np.array(v).T
            if weights.shape[0] <= 1:
                continue
            axs.append(fig.add_subplot(gsc[len(axs), :]))
            axs[-1].imshow(
                    weights_to_img(weights),
                    aspect='auto',
                    interpolation='nearest',
                    cmap='coolwarm')
            axs[-1].set_ylabel(f"{k}"
                               f"\n{np.min(weights):.1e}"
                               f"\n{np.max(weights):.1e}",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)
            if weights.shape[0]:
                axs.append(fig.add_subplot(gsc[len(axs), :]))
                dw = weights[1:] - weights[:-1]

                axs[-1].imshow(
                        weights_to_img(dw),
                        aspect='auto',
                        interpolation='nearest',
                        cmap='coolwarm')
                axs[-1].set_ylabel(f"$d$ {k}"
                                   f"\n{np.min(dw):.1e}"
                                   f"\n{np.max(dw):.1e}",
                                   rotation=0,
                                   labelpad=labelpad,
                                   fontsize=fontsize)
        else:
            continue

    axs[-1].set_xlabel("Iteration", fontsize=fontsize)

    plt.savefig(f"../log/{log_id}/metric.pdf",
                bbox_inches='tight')
    plt.savefig(f"../vis/latest_metric.pdf",
                bbox_inches='tight')

    plt.close()
