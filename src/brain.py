import configparser
import utils


class Brain(object):
    """docstring for Brain"""
    def __init__(self, config):
        """Config loaded from config file"""

        # Properties
        self.size = config.getint('NEURON', 'size')
        self.inputs = config.getint('NEURON', 'inputs')
        self.outputs = config.getint('NEURON', 'outputs')

        # Variables
        self.neurons = utils.initialize_neurons(config=config)
        self.rails = utils.initialize_rails(config=config)

    def update(self):
        pass

    def grow_neurons(self):
        pass

    def prune_neurons(self, idxs=None):
        pass


if __name__ == '__main__':

    # Load config for brain initialization
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Modify brain's config through config file or object calls.
    brain = Brain(config)
    print(brain.neurons)

