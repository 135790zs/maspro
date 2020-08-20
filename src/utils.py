import numpy as np


def normalize(array, lower=0, upper=1):
    # Normalize a [0,1] array to custom range. Does not necessarily include
    # upper and lower bounds.
    array *= upper - lower
    return array + lower


def initialize_neurons(config):
    config = config["NEURON"]
    neurons = dict()
    rng = np.random.default_rng()

    # Number of matrices depends on neuron type
    if config["type"] == "LIF":
        neurons["activation"] = rng.random(size=(config.getint("size")), )
        neurons["activation"] = normalize(
            array=neurons["activation"],
            lower=config.getfloat("min_initial_activation"),
            upper=config.getfloat("max_initial_activation"))

    return neurons

def initialize_rails(config):
    config = config["RAILS"]
    rails = dict()
    rng = np.random.default_rng()



