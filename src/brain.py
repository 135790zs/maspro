import configparser
import utils
import numpy as np
import matplotlib.pyplot as plt


class Brain(object):
    """docstring for Brain"""
    def __init__(self, config):
        """Config loaded from config file"""

        # Properties
        self.time = 0
        self.size = config.getint('size')
        self.inputs = config.getint('inputs')
        self.outputs = config.getint('outputs')

        # Variables
        self.neurons = utils.initialize_neurons(config=config)
        self.rails = utils.initialize_rails(config=config)

        # Visualization initialization
        if config.getboolean("visualize"):
            num_plots_edge = utils.ceiled_sqrt(value=10)
            self.fig, self.axes = plt.subplots(num_plots_edge)
            self.ax = self.axes.ravel()
            plt.ion()
            plt.show()

    def evolve(self):
        self.time += 1

        if config["neurontype"] == "LIF":
            # Leak some from the neurons
            self.neurons["activation"][self.neurons["activation"] < 0] \
                *= config.getfloat("leakage_neg")
            self.neurons["activation"][self.neurons["activation"] > 0] \
                *= config.getfloat("leakage_pos")

            # Perceive inputs
            num_inputs = config.getint("inputs")
            self.neurons["activation"][:num_inputs] = utils.get_inputs(
                time=self.time, num=num_inputs)

            # Find out which units fire
            firing_units = self.neurons["activation"] \
                >= self.neurons["threshold"]

            # Insert the spike into rails
            firing_units_exp = np.repeat(
                a=np.expand_dims(firing_units, axis=0),
                repeats=self.neurons["activation"].size,
                axis=0)
            self.rails["rails"] = utils.insert_2d_in_3d(
                hyperdim=self.rails["rails"],
                hypodim=self.rails["lengths"],
                values=firing_units_exp)

            # Update the activations
            self.neurons["activation"] = \
                np.tanh(self.neurons["activation"]
                        + (np.dot(self.rails["rails"][0, :, :]
                           * self.rails["weights"],
                           self.neurons["activation"])))

            # Evolve trains
            self.rails["rails"] = self.rails["rails"][1:, :, :]
            self.rails["rails"] = np.pad(array=self.rails["rails"],
                                         pad_width=((0, 1), (0, 0), (0, 0)))

        if config["updaterule"] == "mock update":
            self.neurons, self.rails = utils.mock_update(
                neurons=self.neurons,
                rails=self.rails)

        self.plot()

    def plot(self):

        # Remove axes
        for i in range(len(self.ax)):
            self.ax[i].clear()
        ax_count = 0

        plt.suptitle(f"Time = {self.time}")

        self.ax[ax_count].imshow(
            utils.unflatten_to_square(self.rails["weights"]),
            cmap='coolwarm',
            vmin=-1, vmax=1,
            interpolation='nearest')
        self.ax[ax_count].set_title("Weights")
        self.ax[ax_count].axis('tight')
        ax_count += 1

        self.ax[ax_count].imshow(
            utils.unflatten_to_square(self.neurons["activation"]),
            cmap='gray',
            vmin=0, vmax=1,
            interpolation='nearest')
        self.ax[ax_count].set_title("Activations")
        self.ax[ax_count].axis('tight')
        ax_count += 1

        plt.draw()
        plt.savefig('plot.pdf')
        plt.pause(0.001)


if __name__ == '__main__':

    # Load config for brain initialization
    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config["DEFAULT"]

    # Modify brain's config through config file or object calls.
    brain = Brain(config=config)
    for _ in range(10):
        brain.evolve()

# TODO:
"""
v0.3.0. Add test input-output
v0.3.1. Implement STDP
"""
