import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from matplotlib import rcParams as rc
import numpy as np
from graphviz import Digraph
import utils as ut
from config import cfg, lookup

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


def plot_pair(M, t, layers, neurons, fname):
    assert layers[1] - layers[0] in [0, 1]

    plotvars = ["V", "U", "Z", "H", "EVV", "EVU", "DW"]

    fig = plt.figure(constrained_layout=False, figsize=(8, 8))
    gsc = fig.add_gridspec(nrows=len(plotvars) + 2, ncols=1, hspace=0)
    axs = []
    labelpad = 15
    fontsize = 14
    fontsize_legend = 12
    fig.suptitle(f"Epoch {t+1}", fontsize=20)

    # Print global input to network
    axs.append(fig.add_subplot(gsc[len(axs), :],
                               sharex=axs[0] if axs else None))
    axs[-1].plot(M['X'][:t], label="Global")
    axs[-1].plot(M['XZ'][:t], label="Bernoulli")
    axs[-1].set_ylabel(f"$X$",
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)
    axs[-1].legend(fontsize=fontsize_legend,
                   loc="upper right",
                   ncol=2)
    axs[-1].grid(linestyle='--')

    # Print input to neurons
    axs.append(fig.add_subplot(gsc[len(axs), :],
                               sharex=axs[0] if axs else None))
    for n in [0, 1]:
        Z_concat = np.concatenate((M['Z'][:t, layers[n]-1] if layers[n] > 0
                                   else np.pad(np.asarray([M['XZ'][:t]]).T,
                                               ((0, 0), (0,   cfg["N_R"]-1))),
                                   M['Z'][:t, layers[n]]), axis=1)
        into = np.sum(M['W'][:t, layers[n], neurons[n]] * Z_concat, axis=1)
        axs[-1].plot(into, label=f"$x_i$")
        axs[-1].set_ylabel(f"$x$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)
        axs[-1].legend(fontsize=fontsize_legend,
                       loc="upper right",
                       ncol=2)
        axs[-1].grid(linestyle='--')

    for var in plotvars:
        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))
        if M[var][t].ndim == 2:
            axs[-1].plot(M[var][:t, layers[0], neurons[0]],
                         label=f"${lookup[var]['label']}_i$")
            axs[-1].plot(M[var][:t, layers[1], neurons[1]],
                         label=f"${lookup[var]['label']}_j$")
            axs[-1].set_ylabel(f"${lookup[var]['label']}$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        elif M[var][t].ndim == 3:
            EVtype = var[2:]+',' if var[:2] == "EV" else ""
            axs[-1].plot(M[var][:t,
                                layers[1],
                                neurons[1],
                                neurons[0]],
                         label=f"${lookup[var]['label']}_{{{EVtype}i,j}}$")
            if layers[0] == layers[1]:  # Rec, so also plot N1 -> N0
                axs[-1].plot(M[var][:t,
                                    layers[0],
                                    neurons[0],
                                    neurons[1] + cfg["N_R"]],
                             label=f"${lookup[var]['label']}_{{{EVtype}i,j}}$")  # TODO: fix subscript
            axs[-1].set_ylabel(f"${lookup[var]['label']}$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        axs[-1].legend(fontsize=fontsize_legend,
                       loc="upper right",
                       ncol=2)
        axs[-1].grid(linestyle='--')

    axs[-1].set_xlabel("$t$", fontsize=fontsize)

    plt.savefig(f"vis/pair{fname}.pdf",
                bbox_inches='tight')

    plt.close()


def plot_pair_UNUSED(M, t, layers=(0, 1), neurons=(0, 0), X=None, is_unit=False):
    Mt = {}

    for key in lookup:
        if key != "X":
            Mt[key] = M[key]

    Mt_ = {}

    assert layers[1] - layers[0] == 0 or layers[0] - layers[1] == -1
    n1 = neurons[1] + (cfg["N_R"] if layers[0] != layers[1] else 0)
    for key, item in Mt.items():
        if lookup[key]["dim"] == 2:
            if not is_unit:
                Mt_[key] = np.vstack((item[:ep, layers[0], neurons[0]],
                                      item[:ep, layers[1], neurons[1]])).T
            else:
                Mt_[key] = item

        elif lookup[key]["dim"] == 3:
            if not is_unit:
                Mt_[key] = np.vstack((item[:ep, layers[0], neurons[0], n1],  # TODO: Maybe inverse last indices?
                                      item[:ep, layers[1]-1, n1, neurons[0]])).T
            else:
                Mt_[key] = np.vstack((item[:, 0, 1],
                                      item[:, 1, 0])).T
    Mt = Mt_

    fig = plt.figure(constrained_layout=False)
    gsc = fig.add_gridspec(nrows=len(Mt)+int(X is not None), ncols=1, hspace=0)
    axs = []
    labelpad = 15
    fontsize = 14
    fontsize_legend = 12
    fig.suptitle(f"Epoch {ep}"
                 f"{' (' + cfg['neuron'] + ' pair)' if is_unit else ''}",
                 fontsize=20)

    if X is not None:
        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))
        h = 0.5  # height of vlines
        rng = np.random.default_rng()
        inp_ys = rng.random(size=X.shape[0])
        for n_idx in [0, 1]:
            axs[-1].vlines(x=[idx for idx, val in
                              enumerate(X[:, n_idx]) if val],
                           ymin=n_idx+inp_ys/(1+h),
                           ymax=n_idx+(inp_ys+h)/(1+h),
                           colors=f'C{n_idx}',
                           linewidths=0.25,
                           label=f"$x_{n_idx}$")
        axs[-1].set_ylabel("$x^t_j$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)

    for key, arr in Mt.items():
        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))

        if lookup[key]["dim"] == 2:
            axs[-1].plot(arr[:, 0],
                         label=f"${lookup[key]['label']}_0$")
            axs[-1].plot(arr[:, 1],
                         label=f"${lookup[key]['label']}_1$")
            axs[-1].set_ylabel(f"${lookup[key]['label']}_j$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        elif lookup[key]["dim"] == 3:
            EVtype = key[2:]+',' if key[:2] == "EV" else ""
            axs[-1].plot(arr[:, 0],
                         label=f"${lookup[key]['label']}_{{{EVtype}0,1}}$")
            axs[-1].plot(arr[:, 1],
                         label=f"${lookup[key]['label']}_{{{EVtype}1,0}}$")
            axs[-1].set_ylabel(f"${lookup[key]['label']}_{{{EVtype}i,j}}$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        axs[-1].legend(fontsize=fontsize_legend,
                       loc="upper right",
                       ncol=2)
        axs[-1].grid(linestyle='--')

    axs[-1].set_xlabel("$t$", fontsize=fontsize)

    plt.savefig(f"vis/pair{'-sim' if is_unit else ''}.pdf",
                bbox_inches='tight')

    plt.close()


def plot_heatmaps(M, t, fname):

    fig = plt.figure(constrained_layout=False)
    gsc = fig.add_gridspec(nrows=cfg["N_Rec"],
                           ncols=3)
    fig.suptitle(f"Epoch {t+1}", fontsize=20)

    labelpad = 15
    fontsize = 14

    # Neuron heatmaps
    for r in range(0, cfg["N_Rec"]):
        bounds = (-80, 60) if cfg["neuron"] == "Izhikevich" \
            else (0, cfg["thr"])
        axs = fig.add_subplot(gsc[r, 0], label=f"nh-{t}-{r}")
        axs.set_ylabel(f"$v_{{{r}, i}}$",
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)
        axs.imshow(unflatten(M['V'][t, r, :]),
                   cmap='coolwarm',
                   vmin=bounds[0], vmax=bounds[1],
                   interpolation='nearest')

    # Spike heatmap
    for r in range(0, cfg["N_Rec"]):
        axs = fig.add_subplot(gsc[r, 1], label=f"sh-{t}-{r}")
        axs.set_ylabel(f"$z_{{{r}, i}}$",
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)
        axs.imshow(unflatten(M['Z'][t, r, :]),
                   cmap='gray',
                   vmin=0, vmax=1,
                   interpolation='nearest')

    # Weight heatmaps
    for r in range(0, cfg["N_Rec"]):
        maxdev = max(abs(np.min(M['W'])),
                     abs(np.max(M['W'])))
        axs = fig.add_subplot(gsc[r, 2],
                              label=f"wh-{t}-{r}")
        axs.set_ylabel(f"$W_{{{r}, i, j}}$",
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)
        axs.imshow(M['W'][t, r],
                   cmap='bwr',
                   vmin=-maxdev, vmax=maxdev,
                   interpolation='nearest')

    plt.savefig(f"vis/heatmaps{fname}.pdf", bbox_inches='tight')
    plt.close()


def plot_io(M, t, fname):
    fig = plt.figure(constrained_layout=True)
    gsc = fig.add_gridspec(nrows=3, ncols=1, hspace=0.2)
    fig.suptitle(f"Epoch {t}", fontsize=20)

    # Output, target, error
    axs = fig.add_subplot(gsc[0, 0])
    axs.set_title(f"Input + spike")
    axs.plot(M["X"][:t+1])
    axs.plot(M["XZ"][:t+1])

    axs = fig.add_subplot(gsc[1, 0])
    axs.set_title(f"Output")
    axs.plot(np.sum(M['W_out'] * M['Z'][:t+1, -1]))
    axs.plot(M["Y"][:t+1])

    axs = fig.add_subplot(gsc[2, 0])
    axs.set_title(f"Target + error")
    axs.plot(M["T"][:t+1])
    axs.plot(M["error"][:t+1])

    plt.savefig(f"vis/io{fname}.pdf", bbox_inches='tight')
    plt.close()


def plot_drsnn(M, t, layers, neurons, fname):

    if cfg["plot_pair"]:
        plot_pair(fname=fname, M=M, t=t, layers=layers, neurons=neurons)

    if cfg["plot_heatmaps"]:
        plot_heatmaps(fname=fname, M=M, t=t)

    if cfg["plot_io"]:
        plot_io(fname=fname, M=M, t=t)

    if cfg["plot_graph"]:
        plot_graph(fname=fname, M=M, t=t)


def plot_graph(M, t, fname):
    dot = Digraph(format='svg', engine='dot')

    # neurons
    def neuroncolor(r, n, spiked):
        bounds = (-80, 60) if cfg["neuron"] == "Izhikevich" \
            else (0, cfg["thr"])
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
            # if (r == 0 and n >= cfg["N_I"]) or \
            #    (r == cfg["N_Rec"]-1 and n >= cfg["N_O"]):
            #     continue
            spiked = bool(M['Z'][t, r, n])
            dot.node(name=f"{r}-{n}",
                     label=f"{r}-{n}\n{M['V'][t, r, n]:.2f}",
                     style='radial',
                     fixedsize='false',
                     color="#ffffff",
                     fillcolor=neuroncolor(r=r, n=n, spiked=spiked))

    # weights
    def weightcolor(w):
        maxdev = max(abs(np.min(M['W'])),
                     abs(np.max(M['W'])))
        w = (w / maxdev) + (1 / 2)
        cmap = mpcm.get_cmap("bwr")
        rgba = cmap(w, bytes=True)
        ret = "#"
        for val in rgba:
            ret += hex(val)[2:]
        return ret

    # in-to-rec
    for head in range(0, cfg["N_R"]):
        dot.edge(tail_name=f"in",
                 head_name=f"{0}-{head}",
                 label=f"{M['W'][t, 0, 0, head]:.2f}",
                 penwidth='1',
                 color=weightcolor(w=M['W'][t, 0, 0, head]))

    # intra-rec
    for r in range(0, cfg["N_Rec"]):  # r to r+1
        for head in range(0, cfg["N_R"]):
            for tail in range(cfg["N_R"], 2*cfg["N_R"]):
                if M['W'][t, r, head, tail] != 0:
                    dot.edge(tail_name=f"{r}-{tail-cfg['N_R']}",
                             head_name=f"{r}-{head}",
                             label=f"{M['W'][t, r, head, tail]:.2f}",
                             penwidth='1',
                             color=weightcolor(w=M['W'][t, r, head, tail]))
    # inter-rec
    for r in range(0, cfg["N_Rec"]-1):  # r to r+1
        for head in range(0, cfg["N_R"]):
            for tail in range(0, cfg["N_R"]):
                if M['W'][t, r, head, tail] != 0:
                    dot.edge(tail_name=f"{r}-{head}",
                             head_name=f"{r+1}-{tail}",
                             label=f"{M['W'][t, r, head, tail]:.2f}",
                             penwidth='1',
                             color=weightcolor(w=M['W'][t, r, head, tail]))
    # rec-to-out
    for tail in range(0, cfg["N_R"]):
        dot.edge(tail_name=f"{cfg['N_Rec']-1}-{tail}",
                 head_name=f"out",
                 label=f"{M['W_out'][tail]:.2f}",
                 penwidth='1',
                 color=weightcolor(w=M['W_out'][tail]))

    dot.attr(label=f"Epoch {t+1}")
    dot.render(f"vis/net{fname}")


def unflatten(arr):
    a = arr.flatten().shape[0]
    b = int(np.ceil(np.sqrt(a)))
    arr = np.concatenate((arr.flatten(), np.zeros(b*b-a)))
    arr = np.reshape(arr, (b, b))
    c = b
    while (b*c-a) >= b:
        c -= 1
        arr = arr[:-1, :]
    return arr
