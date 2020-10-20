import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from matplotlib import rcParams as rc
import numpy as np
from graphviz import Digraph
import utils as ut
from config import cfg, lookup

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


def plot_pair(M, ep, layers=(0, 1), neurons=(0, 0), X=None, is_unit=False):
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
                Mt_[key] = Mt[key]

        elif lookup[key]["dim"] == 3:
            if not is_unit:
                Mt_[key] = np.vstack((item[:ep, layers[0], neurons[0], n1],
                                      item[:ep, layers[1], n1, neurons[0]])).T
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
    fig.suptitle(f"Epoch {ep} ({cfg['neuron'] + ' pair' if is_unit else ''})",
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

    plt.savefig(f"vis/pair{'-sim' if is_unit else ''}.pdf")


def plot_heatmaps(M, ep):

    fig = plt.figure(constrained_layout=False)
    gsc = fig.add_gridspec(nrows=cfg["N_Rec"],
                           ncols=3)
    fig.suptitle(f"Epoch {ep}", fontsize=20)

    labelpad = 15
    fontsize = 14

    # Neuron heatmaps
    for r in range(0, cfg["N_Rec"]):
        num = cfg["N_I"] if r == 0 \
            else cfg["N_O"] if r == cfg["N_Rec"] \
            else cfg["N_R"]
        axs = fig.add_subplot(gsc[r, 0], label=f"nh-{ep}-{r}")
        axs.axis('off')
        axs.set_ylabel(f"$v_{{{r}, i}}$",
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)
        axs.imshow(unflatten(M['V'][ep, r, :num]),
                   cmap='coolwarm',
                   vmin=-85, vmax=cfg["eqb"],
                   interpolation='nearest')

    # Neuron spikes
    for r in range(0, cfg["N_Rec"]):
        num = cfg["N_I"] if r == 0 \
            else cfg["N_O"] if r == cfg["N_Rec"] \
            else cfg["N_R"]
        axs = fig.add_subplot(gsc[r, 1], label=f"ns-{ep}-{r}")
        axs.axis('off')
        axs.set_ylabel(f"$z_{{{r}, i}}$",
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)
        axs.imshow(unflatten(M['Z'][ep, r, :num]),
                   cmap='gray',
                   vmin=0, vmax=1,
                   interpolation='nearest')

    # Weight heatmaps
    for r in range(0, cfg["N_Rec"]-1):
        axs = fig.add_subplot(gsc[r, 2],
                              label=f"wh-{ep}-{r}")
        axs.axis('off')
        axs.set_ylabel(f"$W_{{{r}, i, j}}$",
                       rotation=0,
                       labelpad=labelpad,
                       fontsize=fontsize)
        axs.imshow(M['W'][ep, r, :, :cfg["N_R"]],
                   cmap='coolwarm',
                   vmin=0, vmax=100,
                   interpolation='nearest')
    plt.savefig("vis/heatmaps.pdf", bbox_inches='tight')
    plt.close()


def plot_io(M, ep):
    fig = plt.figure(constrained_layout=True)
    gsc = fig.add_gridspec(nrows=2, ncols=2, hspace=0.2)
    fig.suptitle(f"Epoch {ep}", fontsize=20)

    # Output, target, error
    axs = fig.add_subplot(gsc[0, 0])
    axs.set_title(f"Input + spike")
    axs.plot(M["input"][:ep+1, :])
    axs.plot(M["input_spike"][:ep+1, :])

    axs = fig.add_subplot(gsc[0, 1])
    axs.set_title(f"Output + EMA")
    axs.plot(M["output"][:ep+1, :])
    axs.plot(M["output_EMA"][:ep+1, :])

    axs = fig.add_subplot(gsc[1, 0])
    axs.set_title(f"Target + EMA")
    axs.plot(M["target"][:ep+1, :])
    axs.plot(M["target_EMA"][:ep+1, :])

    axs = fig.add_subplot(gsc[1, 1])
    axs.set_title(f"Error + EMA")
    axs.plot(ut.errfn(M["target"][:ep+1, :], M["output"][:ep+1, :]))
    axs.plot(ut.errfn(M["target_EMA"][:ep+1, :], M["output_EMA"][:ep+1, :]))
    plt.savefig("vis/io.pdf", bbox_inches='tight')
    plt.close()


def plot_drsnn(M, ep, layers=(0, 1), neurons=(0, 0)):

    if cfg["plot_pair"]:
        plot_pair(M=M, ep=ep, layers=layers, neurons=neurons)

    if cfg["plot_heatmaps"]:
        plot_heatmaps(M=M, ep=ep)

    if cfg["plot_io"]:
        plot_io(M=M, ep=ep)

    if cfg["plot_graph"]:
        plot_graph(M=M, ep=ep)


def plot_graph(M, ep):
    dot = Digraph(format='svg', engine='dot')

    # neurons
    def neuroncolor(r, n, spiked):
        bounds = (-80, 60) if cfg["neuron"] == "Izhikevich" else (-80, 60)
        v = (M['V'][ep, r, n] - bounds[0]) / bounds[1]
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
            if (r == 0 and n >= cfg["N_I"]) or \
               (r == cfg["N_Rec"]-1 and n >= cfg["N_O"]):
                continue
            spiked = bool(M['Z'][ep, r, n])
            dot.node(name=f"{r}-{n}",
                     style='radial',
                     fixedsize='false',
                     color="#ffffff",
                     fillcolor=neuroncolor(r=r, n=n, spiked=spiked))

    # weights
    def weightcolor(w):
        w = (w - np.min(M['W'])) / np.max(M['W'])
        cmap = mpcm.get_cmap("bwr")
        rgba = cmap(w, bytes=True)
        ret = "#"
        for val in rgba:
            ret += hex(val)[2:]
        return ret

    for r in range(0, cfg["N_Rec"]-1):
        for n1 in range(0, cfg["N_R"]):
            for n2 in range(0, cfg["N_R"]*2):

                n2n = n2 if n2 < cfg["N_R"] else n2 % cfg["N_R"]
                rn = r if n2 < cfg["N_R"] else r + 1

                if (r == 0 and n1 >= cfg["N_I"]) or \
                   (r == cfg["N_Rec"]-2 and n2n >= cfg["N_O"]):
                    continue

                if M['W'][ep, r, n2, n1] != 0:
                    dot.edge(tail_name=f"{r}-{n1}",
                             head_name=f"{rn}-{n2n}",
                             penwidth='1',
                             color=weightcolor(w=M['W'][ep, r, n2, n1]))

    dot.attr(label=f"Epoch {ep}")
    dot.render("vis/net")


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
