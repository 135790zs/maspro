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
    # n_nodes = 6
    # neurons = np.random.random(size=(n_nodes,))
    # weights = np.random.random(size=(n_nodes, n_nodes))
    # # weights = np.asarray([[.1,.2,.5], [.7, .8,.5], [.1, 0, 0]])

    # lengths = np.random.randint(low=2, high=8, size=(n_nodes, n_nodes))

    dot = Digraph(format='png', engine='neato')
    print(neurons["activation"])

    for idx, node in enumerate(neurons["activation"]):
        pos = str(hex(max(int(255*node), int(255*minvis))))[2:]
        dot.node(name=str(idx), 
                 label=f"{node:.2f}", 
                 color=f"#000000{pos}")
    # edges
    for idx, start in enumerate(rails["weights"]):
        for jdx, end in enumerate(start):
            if end:# and idx != jdx:
                end = end / 2 + 0.5
                pos = str(hex(max(int(255*end), int(255*minvis))))[2:]
                dot.edge(tail_name=str(idx), 
                         head_name=str(jdx), 
                         color=f"#000000{pos}",
                         len=str(rails["lengths"][idx][jdx]/50))

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

    if config["topology"] == "full":
        rails["weights"] = rng.random(
            size=(config.getint("size"), ) * 2)

    rails["lengths"] = rng.integers(low=config.getint("min_rails_length"),
                                    high=config.getint("max_rails_length"),
                                    size=rails["weights"].shape, )

    rails["rails"] = np.zeros(shape=(config.getint("max_rails_length"),
                                     *rails["weights"].shape, ))
    return rails

