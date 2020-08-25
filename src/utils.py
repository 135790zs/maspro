import numpy as np
from graphviz import Digraph


def insert_2d_in_3d(hyperdim, hypodim, values=1):
    size = hyperdim.shape[-1]  # Assume inner two shapes are equal
    reprange = np.repeat(np.expand_dims(np.arange(size), axis=0), size, axis=0)
    hyperdim[hypodim, reprange.T, reprange] = values
    return hyperdim


def ceiled_sqrt(value):
    return int(np.ceil(np.sqrt(value)))


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


def mock_update(neurons, rails):
    return neurons, rails


def draw_graph(neurons, rails, fname, minvis=.2):

    dot = Digraph(format='png', engine='neato')

    # nodes
    for idx, node in enumerate(neurons["activation"]):
        firing = node >= neurons["threshold"][idx]
        firing_red = 'aa' if firing else '00'
        pos = str(hex(max(int(255 * node),
                          int(255 * minvis))))[2:]
        dot.node(name=str(idx),
                 label=f"{node:.2f}",
                 color=f"#{firing_red}0000{pos}")

    # edges
    for idx_end, end in enumerate(rails["weights"]):
        for idx_start, weight in enumerate(end):
            if weight:

                # Rail weight determines edge transparency
                weight_alph = weight / 2 + 0.5
                weight_alph = str(hex(max(int(255 * weight_alph),
                                          int(255 * minvis))))[2:]

                # Choose the redness for edges, determined by num of spikes
                has_spike_sum = np.sum(rails["rails"][:, idx_end, idx_start])
                has_spike = has_spike_sum / rails["lengths"][idx_end, idx_start]
                spike_red = str(hex(int(255 * has_spike)))[2:]
                spike_red = '00' if spike_red == '0' else spike_red

                dot.edge(tail_name=str(idx_start),
                         head_name=str(idx_end),
                         color=f"#{spike_red}0000{weight_alph}",
                         len=str(max(1, rails["lengths"][idx_end, idx_start] / 50)))

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

    return neurons


def initialize_rails(config):
    rails = dict()
    rng = np.random.default_rng()

    # Initialize connections. Zero-weights = unconnected.
    # Inner = from, outer = to.
    if config["topology"] == "full":
        rails["weights"] = rng.random(
            size=(config.getint("size"), ) * 2)

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

