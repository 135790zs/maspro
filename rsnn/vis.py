import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from matplotlib import rcParams as rc
import numpy as np
from graphviz import Digraph
from config import cfg, lookup

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'


def plot_run(terrs, verrs, W, epoch):
    labelpad = 35
    fontsize = 14
    fig = plt.figure(constrained_layout=False, figsize=(8, 8))
    gsc = fig.add_gridspec(nrows=5, ncols=1, hspace=0.05)
    axs = []
    for errs, label in [(terrs, "E_T"), (verrs, "E_V")]:
        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))
        axs[-1].plot(errs[:epoch])
        axs[-1].grid()
        axs[-1].set_ylabel(f"${label}$",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)
    if epoch >= 1:
        for weight_type in ['W', 'W_out', 'b_out']:
            # W
            axs.append(fig.add_subplot(gsc[len(axs), :], sharex=axs[0]))
            axs[-1].imshow(
                W[weight_type][:epoch].reshape(
                    W[weight_type][:epoch].shape[0], -1).T,
                aspect='auto',
                interpolation='nearest',
                cmap='coolwarm')
            axs[-1].set_ylabel(f"{weight_type}"
                               f"\n[{np.min(W[weight_type][:epoch]):.1f}"
                               f", {np.max(W[weight_type][:epoch]):.1f}]",
                               rotation=0,
                               labelpad=labelpad,
                               fontsize=fontsize)

    plt.savefig(f"../vis/errs.pdf",
                bbox_inches='tight')

    plt.close()


def plot_state(M, W_rec, W_out, b_out):
    M_plotvars = ["X", "I", "V", "H", "U", "Z_in", "Z", "ZbarK", "EVV", "EVU",
                  "ET", "ETbar", "Y", "P", "Pmax", "T", "CE", "DW", "DW_out", "Db_out"]
    W = {"W_rec": W_rec, "W_out": W_out, "b_out": b_out}

    fig = plt.figure(constrained_layout=False, figsize=(8, 14))
    gsc = fig.add_gridspec(nrows=len(M_plotvars) + len(W.keys()) + 2,
                           ncols=1,
                           hspace=0.075)
    axs = []
    labelpad = 35
    fontsize = 14
    fig.suptitle(f"Single-run model state\n$\\alpha={cfg['alpha']:.3f}$, "
                 f"$\\kappa={cfg['kappa']:.3f}$, $\\rho={cfg['rho']:.3f}$",
                 fontsize=20)

    for var in M_plotvars:

        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))
        if lookup[var]['scalar'] == False:
            axs[-1].imshow(M[var].reshape(M[var].shape[0], -1).T,
                           cmap=('gray' if lookup[var]["binary"]
                                 else 'coolwarm'),
                           vmin=np.min(M[var]),
                           vmax=np.max(M[var]),
                           interpolation='nearest',
                           aspect='auto')
        else:
            axs[-1].plot(M[var])
            axs[-1].grid()

        axs[-1].set_ylabel(f"${lookup[var]['label']}$"
                           f"\n[{np.min(M[var]):.1f}"
                           f", {np.max(M[var]):.1f}]",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)
    for k, v in W.items():
        v = v.flatten()
        v = np.expand_dims(v, axis=0)
        v = np.repeat(v, M['X'].shape[0], axis=0)

        axs.append(fig.add_subplot(gsc[len(axs), :],
                                   sharex=axs[0] if axs else None))
        axs[-1].imshow(v.reshape(v.shape[0], -1).T,
                       cmap='coolwarm',
                       vmin=np.min(v),
                       vmax=np.max(v),
                       interpolation='nearest',
                       aspect='auto')
        axs[-1].set_ylabel(f"${lookup[k]['label']}$"
                           f"\n[{np.min(v):.1f}"
                           f", {np.max(v):.1f}]",
                           rotation=0,
                           labelpad=labelpad,
                           fontsize=fontsize)


    axs[-1].set_xlabel("$t$", fontsize=fontsize)

    plt.savefig(f"../vis/state.pdf",
                bbox_inches='tight')

    plt.close()


def plot_graph(M, t, W_rec, W_out):
    dot = Digraph(format='svg', engine='dot')

    # neurons
    def neuroncolor(r, n, spiked):
        if r is None:
            return None

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

    for n in range(0, cfg["N_I"]):
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
    for tail in range(0, cfg["N_I"]):
        for head in range(0, cfg["N_R"]):
            dot.edge(tail_name=f"in-{tail}",
                     head_name=f"{0}-{head}",
                     label=f"{W_rec[0, head, tail]:.2f}",
                     penwidth='1',
                     color=weightcolor(w=W_rec[0, head, tail]))

    # intra-rec
    for r in range(0, cfg["N_Rec"]):  # r to r+1
        for head in range(0, cfg["N_R"]):
            for tail in range(cfg["N_R"], 2*cfg["N_R"]):
                if W_rec[r, head, tail] != 0:
                    dot.edge(tail_name=f"{r}-{tail-cfg['N_R']}",
                             head_name=f"{r}-{head}",
                             label=f"{W_rec[r, head, tail]:.2f}",
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
        for head in range(0, cfg["N_O"]):
            dot.edge(tail_name=f"{cfg['N_Rec']-1}-{tail}",
                     head_name=f"out-{head}",
                     label=f"{W_out[tail, head]:.2f}",
                     penwidth='1',
                     color=weightcolor(w=W_out[tail, head]))

    dot.attr(label=f"Steps {t+1}")
    dot.render(f"../vis/net")
