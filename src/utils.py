import numpy as np
from graphviz import Digraph


def insert_2d_in_3d(hyperdim, hypodim, values=1):
    size = hyperdim.shape[-1]  # Assume inner two shapes are equal
    reprange = np.repeat(np.expand_dims(np.arange(size), axis=0), size, axis=0)
    hyperdim[hypodim, reprange.T, reprange] = values
    return hyperdim


def ceiled_sqrt(value):
    return int(np.ceil(np.sqrt(value)))


def synaptic_scaling(weights, factor):
    """ Weight is weight times the scaled average of the in-syns."""
    for neuron_idx, in_syn in enumerate(weights):
        n_nonzero = np.count_nonzero(in_syn)
        if n_nonzero:
            weights[neuron_idx] = in_syn * factor * n_nonzero / np.sum(in_syn)

    return weights


def unflatten_to_square(arr):
    size = arr.flatten().shape[0]
    if size <= 0:
        raise ValueError("Could not unflatten empty array.")
    edge = ceiled_sqrt(size)
    arr = np.concatenate((arr.flatten(), np.zeros(edge * edge - size)))
    arr = np.reshape(arr, (edge, edge))
    count = edge
    while (edge * count - size) >= edge:
        count -= 1
        arr = arr[:-1, :]
    return arr


def normalize(array, lower=0, upper=1):
    # Normalize a [0,1] array to custom range. Does not necessarily include
    # upper and lower bounds.
    array *= upper - lower
    return array + lower


def stdp(neurons, rails, dopa=None, metaplas=None):
    """ Presyn before postsyn = presyn.trace < postsyn.trace = increase"""
    # for all traces 1, update with all other if weight != 0.
    plasticity = list()
    for idx_neuron, trace in enumerate(neurons["trace"]):
        if trace == 1:  # just fired
            for idx_other, weight in enumerate(
                    rails["weights"][:, idx_neuron]):
                trace_other = neurons["trace"][idx_other]
                if weight != 0 and trace_other > 0 and trace_other < 1:

                    if dopa:
                        trace_other *= dopa

                    if metaplas:
                        trace_other *= metaplas

                    factor = 1 / (1 - trace_other)

                    plasticity.append(factor)

                    rails["weights"][idx_other, idx_neuron] /= factor  # pre to post
                    rails["weights"][idx_neuron, idx_other] *= factor  # post to pre
    plasticity = 1 / np.mean(plasticity) if plasticity else 1
    return neurons, rails, plasticity


def update(neurons, rails, config, **kwargs):
    update_rule = config["update_rule"]
    if update_rule == "none":
        return neurons, rails
    elif update_rule == "stdp":
        return stdp(neurons=neurons, rails=rails, dopa=kwargs["dopa"])


def draw_graph(neurons, rails, fname, config, minvis=.2):

    dot = Digraph(format='png', engine='neato')

    # nodes
    for idx, node in enumerate(neurons["activation"]):
        firing = node >= neurons["threshold"][idx]
        firing_red = 'ff' if firing else '00'

        transparency = str(hex(max(int(255 * node),
                                   int(255 * minvis))))[2:]

        input_yel = 'ff' if idx < config.getint("inputs") else 'bb'
        output_cyan = 'ff' if idx >= config.getint("size") - \
            config.getint("outputs") else 'bb'

        dot.node(name=str(idx),
                 style='filled',
                 fixedsize='false',
                 fillcolor=f"#{input_yel}ff{output_cyan}",
                 color=f"#{firing_red}0000{transparency}")

    # edges
    for idx_end, end in enumerate(rails["weights"]):
        for idx_start, weight in enumerate(end):
            if weight:

                # Rail weight determines edge transparency
                weight_alph = weight - np.min(rails["weights"])
                weight_alph = weight_alph / np.max(rails["weights"])
                # weight *= np.max(rails["weights"]) /
                weight_alph = str(hex(max(int(255 * weight_alph),
                                          int(255 * minvis))))[2:]

                # Choose the redness for edges, determined by num of spikes
                # has_spike_sum = np.sum(rails["rails"][:, idx_end, idx_start])
                # has_spike = has_spike_sum / rails["lengths"][idx_end, idx_start]
                # spike_red = str(hex(int(255 * has_spike)))[2:]
                # spike_red = '00' if spike_red == '0' else spike_red

                collist = str()
                this_rail_len = rails["lengths"][idx_end, idx_start]
                floored = int(1/this_rail_len*100)/100
                for rail in range(this_rail_len):
                    if rails["rails"][rail, idx_end, idx_start]:
                        collist = f"#ff3333{weight_alph};{floored:.2f}:" + collist
                    else:
                        collist = f"#333333{weight_alph};{floored:.2f}:" + collist
                dot.edge(tail_name=str(idx_start),
                         head_name=str(idx_end),
                         penwidth='3',
                         color=collist,
                         # color=f"#{spike_red}0000{weight_alph}",
                         len=str(max(1, rails["lengths"][idx_end, idx_start] / 10)))

    dot.render(fname)


def initialize_neurons(config):
    neurons = dict()
    rng = np.random.default_rng()

    # Number of matrices depends on neuron type
    if config["neurontype"] == "LIF":
        neurons["activation"] = rng.random(size=(config.getint("size")), )
        neurons["activation"] = normalize(
            array=neurons["activation"],
            lower=config.getfloat("min_initial_activation"),
            upper=config.getfloat("max_initial_activation"))

        neurons["threshold"] = rng.random(size=neurons["activation"].shape)
        neurons["threshold"] = normalize(
            array=neurons["threshold"],
            lower=config.getfloat("min_initial_threshold"),
            upper=config.getfloat("max_initial_threshold"))
        neurons["trace"] = np.zeros(shape=neurons["activation"].shape)

    return neurons


def initialize_rails(config):
    rails = dict()
    rng = np.random.default_rng()
    n_nodes = config.getint("size")
    # Initialize connections. Zero-weights = unconnected.
    # Inner = from, outer = to.
    rails["weights"] = rng.random(
        size=(n_nodes, ) * 2)

    rails["weights"] = normalize(rails["weights"],
                                 lower=config.getfloat("minweight"),
                                 upper=config.getfloat("maxweight"))

    if config["topology"] == "lattice":

        assert n_nodes % 2 == 0, "When using a lattice topology, use an even" \
                                 " number of neurons. "

        adjacency = np.zeros(shape=rails["weights"].shape, dtype=bool)
        sqt = int(np.sqrt(n_nodes))
        for v in range(n_nodes):
            if (v+1) % sqt:
                adjacency[v, v + 1] = True
                adjacency[v + 1, v] = True
            if v % sqt:
                adjacency[v, v - 1] = True
                adjacency[v - 1, v] = True
            if v + 1 + sqt <= n_nodes:
                adjacency[v + sqt, v] = True
                adjacency[v, v + sqt] = True
            if v + 1 - sqt > 0:
                adjacency[v - sqt, v] = True
                adjacency[v, v - sqt] = True
        rails["weights"][adjacency == False] = 0

    indices = np.random.choice(np.arange(rails["weights"].size),
                               replace=False,
                               size=int(rails["weights"].size
                                        * config.getfloat('dropout')))
    rails["weights"][np.unravel_index(indices, rails["weights"].shape)] = 0

    # Nullify weights to input neurons
    rails["weights"][:config.getint("inputs"), :] = 0

    # Nullify weights from output neurons
    rails["weights"][:, -config.getint("outputs"):] = 0

    # Initialize rail lengths
    rails["lengths"] = rng.integers(low=config.getint("min_rails_length"),
                                    high=config.getint("max_rails_length"),
                                    size=rails["weights"].shape, )

    rails["rails"] = np.zeros(shape=(config.getint("max_rails_length"),
                                     *rails["weights"].shape, ))
    return rails
