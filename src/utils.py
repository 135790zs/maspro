import numpy as np


def insert_2d_in_3d(hyperdim, hypodim, values=1):
    size = hyperdim.shape[-1]  # Assume inner two shapes are equal
    reprange = np.repeat(np.expand_dims(np.arange(size), axis=0), size, axis=0)
    hyperdim[hypodim, reprange.T, reprange] = values
    return hyperdim


def get_inputs(time, num):
    return range(num)


def ceiled_sqrt(value):
    return int(np.ceil(np.sqrt(value)))


def unflatten_to_square(arr):
    size = arr.flatten().shape[0]
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

