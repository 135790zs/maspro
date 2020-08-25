import configparser
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io_functions


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

        # Logging
        self.log_input = np.zeros(shape=(self.inputs, 0))
        self.log_input_poisson = np.zeros(shape=self.log_input.shape)
        self.log_output = np.zeros(shape=(self.outputs, 0))

        # Visualization initialization
        if config.getboolean("visualize"):
            num_plots_edge = utils.ceiled_sqrt(value=6)
            self.fig, self.axes = plt.subplots(num_plots_edge, num_plots_edge)
            self.ax = self.axes.ravel()
            plt.ion()
            plt.show()

    def evolve(self):
        self.time += 1

        if config["neurontype"] == "LIF":
            # Leak, input, fire, integrate, update-rails

            # Leak some from the neurons
            self.neurons["activation"][self.neurons["activation"] < 0] \
                *= config.getfloat("leakage_neg")
            self.neurons["activation"][self.neurons["activation"] > 0] \
                *= config.getfloat("leakage_pos")

            self.neurons["trace"] /= 2
            # Add fire to eli trace

            # Find out which units fire
            firing_units = self.neurons["activation"] \
                >= self.neurons["threshold"]

            # Insert the spike into rails
            firing_units_exp = np.repeat(
                a=np.expand_dims(firing_units, axis=0),
                repeats=self.neurons["activation"].size,
                axis=0)

            self.neurons["trace"][firing_units] = 1

            self.rails["rails"] = utils.insert_2d_in_3d(
                hyperdim=self.rails["rails"],
                hypodim=self.rails["lengths"],
                values=firing_units_exp)

            # Reset firing units
            self.neurons["activation"][firing_units] = config.getfloat("reset")

            # Update the activations
            self.neurons["activation"] = \
                np.tanh(self.neurons["activation"]
                        + np.sum(self.rails["rails"][0, :, :]
                                 * self.rails["weights"],
                                 axis=1))

            # Evolve trains
            self.rails["rails"] = self.rails["rails"][1:, :, :]
            self.rails["rails"] = np.pad(array=self.rails["rails"],
                                         pad_width=((0, 1), (0, 0), (0, 0)))

            # Perceive inputs
            num_inputs = config.getint("inputs")
            new_input = io_functions.istream(time=self.time)

            rng = np.random.default_rng()
            self.neurons["activation"][:num_inputs] = \
                [rng.binomial(n=1, p=inp) for inp in new_input]

            # Logging
            self.log_input = np.append(self.log_input, new_input)
            self.log_input_poisson = np.append(
                self.log_input_poisson,
                self.neurons["activation"][:num_inputs])
            self.log_output = np.append(
                self.log_output,
                self.neurons["activation"][config.getint("outputs"):])

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

        def subplot(title, array, ax_count, vmin=0, vmax=1, hmap=None):
            self.ax[ax_count].set_title(title)
            self.ax[ax_count].axis('tight')

            if len(array.shape) > 1 or hmap:
                heatmap(array=array, ax_count=ax_count, vmin=vmin, vmax=vmax)
            else:
                lineplot(array=array, ax_count=ax_count, vmin=vmin, vmax=vmax)
            return ax_count + 1

        def heatmap(array, ax_count, vmin=0, vmax=1):
            self.ax[ax_count].imshow(
                utils.unflatten_to_square(array),
                cmap='coolwarm' if abs(vmin) == abs(vmax) else 'gray',
                vmin=vmin, vmax=vmax,
                interpolation='nearest')

        def lineplot(array, ax_count, vmin=0, vmax=1):
            self.ax[ax_count].plot(array)
            self.ax[ax_count].set_ylim(vmin, vmax)

        ax_count = subplot(array=self.rails["weights"],
                           title="Weights",
                           ax_count=ax_count,
                           vmin=-1,
                           vmax=1)
        ax_count = subplot(array=self.neurons["activation"],
                           title="Activation",
                           ax_count=ax_count,
                           vmin=0,
                           vmax=1,
                           hmap=True)
        ax_count = subplot(array=self.log_input,
                           title="Input source",
                           ax_count=ax_count)
        ax_count = subplot(array=self.log_input_poisson,
                           title="Input spikes",
                           ax_count=ax_count)
        ax_count = subplot(array=self.log_output,
                           title="Output",
                           ax_count=ax_count)

        # Spiketrains
        ev = self.rails["rails"]
        ev = np.reshape(ev, (ev.shape[0], ev.shape[1]*ev.shape[2]), order='F').T
        for t in range(ev.shape[1]):
            ev[:, t] *= t + 1
        self.ax[ax_count].eventplot(ev, linelengths=0.3, alpha=0.5)
        self.ax[ax_count].set_xlim(0, ev.shape[1])
        self.ax[ax_count].set_title("Spike train")

        # also show entry points
        lens = np.zeros(shape=ev.shape)
        all_lengths = self.rails["lengths"].flatten(order='F')
        for idx, L in enumerate(lens):
            lens[idx, all_lengths[idx]] = 1
        for ln in range(lens.shape[1]):
            lens[:, ln] *= ln + 1
        self.ax[ax_count].eventplot(lens, color='red', alpha=0.5)
        ax_count += 1

        # Graph
        fname = "graph"
        utils.draw_graph(self.neurons, self.rails, fname=fname)
        img1 = mpimg.imread(fname + ".png")
        self.ax[ax_count].imshow(img1)
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

    n_epochs = -1
    while brain.time != n_epochs:
        brain.evolve()

"""
v0.2.3: Add model graph in visualization.
v0.2.4: Improve edge and node color; nullify weights to input and from output.
v0.3.0. Add test input-output; variable weight range; improve vis; change activation function from dot to sum.

TODO MAJOR:
v0.3.1. Implement STDP

TODO MINOR:
* Export `evolve' and `plot' to function, try decouple

"""
