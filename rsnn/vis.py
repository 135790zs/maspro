import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from matplotlib import rcParams as rc
import numpy as np
from graphviz import Digraph
from config import cfg, lookup

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


def plot_state(M, t, fname, layers=None, neurons=None):
    print(t)
    plotvars = ["X", "XZ", "I", "V", "U", "Z", "H", "EVV", "EVU", "ET", "DW", "W",
                "Y", "T", "error"]

    fig = plt.figure(constrained_layout=False, figsize=(10, 14))
    gsc = fig.add_gridspec(nrows=len(plotvars) + 2, ncols=1, hspace=0)
    axs = []
    labelpad = 15
    fontsize = 14
    fontsize_legend = 12
    fig.suptitle(f"Epoch {t+1}", fontsize=20)

    # Print input to neurons
    for var in plotvars:
        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))
        if M[var][t].ndim == 0:  # Y, T, error
            axs[-1].plot(M[var][:t])
            axs[-1].set_ylabel(var,
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)
        if M[var][t].ndim == 1:  # X, XZ
            axs[-1].plot(M[var][:t],)
            axs[-1].set_ylabel(var,
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)
        if M[var][t].ndim == 2:  # information per neuron
            if layers is not None:
                axs[-1].plot(M[var][:t, layers[0], neurons[0]],
                             label=f"${lookup[var]['label']}_i$")
                axs[-1].plot(M[var][:t, layers[1], neurons[1]],
                             label=f"${lookup[var]['label']}_j$")
            else:
                axs[-1].imshow(M[var][:t].reshape(t, -1).T,
                               cmap='coolwarm',
                               vmin=np.min(M[var][:t]),
                               vmax=np.max(M[var][:t]),
                               interpolation='nearest',
                               aspect='auto')
            axs[-1].set_ylabel(f"${lookup[var]['label']}$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

        elif M[var][t].ndim == 3:  # information per weight
            if layers is not None:
                axs[-1].plot(M[var][:t,
                                    layers[1],
                                    neurons[1],
                                    neurons[0]],
                             label=f"${lookup[var]['label']}$")
                if layers[0] == layers[1]:  # Rec, so also plot N1 -> N0
                    axs[-1].plot(M[var][:t,
                                        layers[0],
                                        neurons[0],
                                        neurons[1] + cfg["N_R"]],
                                 label=f"${lookup[var]['label']}$")
            else:
                axs[-1].imshow(M[var][:t].reshape(t, -1).T,
                               cmap='coolwarm',
                               vmin=np.min(M[var][:t]),
                               vmax=np.max(M[var][:t]),
                               interpolation='nearest',
                               aspect='auto')
            axs[-1].set_ylabel(f"${lookup[var]['label']}$",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)
        if layers is not None or M[var][t].ndim < 2:
            # axs[-1].legend(fontsize=fontsize_legend,
            #                loc="upper right",
            #                ncol=2)
            axs[-1].grid(linestyle='--')

    axs[-1].set_xlabel("$t$", fontsize=fontsize)

    plt.savefig(f"vis/state{fname}.pdf",
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


def plot_drsnn(M, t, fname="", layers=None, neurons=None):

    if cfg["plot_state"]:
        plot_state(fname=fname, M=M, t=t, layers=layers, neurons=neurons)

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
