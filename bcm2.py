import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def inp(n):
    val = 1 if n % 10 == 3 else 0
    return np.asarray([val])
    # return np.asarray([.2 + .2 * np.sin(n * 0.3)])
    # return np.asarray([])


def out(n):
    # return np.asarray([.1 + .1 * np.sin(n * 0.3)])
    return np.asarray([.6])


def f(x):
    return np.tanh(x)


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def insert_2d_in_3d(M, a, val=1):
    N = M.shape[-1]
    R = np.repeat(np.expand_dims(np.arange(N), axis=0), N, axis=0)
    M[a, R.T, R] = val
    return M


class ESN():
    def __init__(self, inp, size, out, topo, dropout, weight_scaling, show_plots=False):
        np.random.seed()
        self.time = 0
        self.plot_res = 1
        self.spiking = False
        self.plot_reach = int(1e3)
        self.inp = inp
        self.inp_size = inp(0).size
        self.out = out
        self.out_size = out(0).size
        self.size = size

        self.threshold = 0.3
        self.reset = -.2

        self.noise_factor_w = 0.001
        self.noise_factor_a = 0.
        self.lr = 2
        self.leakage_down = .99
        self.leakage_up = 0.7
        self.P = 1
        self.P_current_weight = 1

        self.L = np.zeros(shape=(size, size))  # TODO: May be removed possibly
        self.L_current_weight = 10

        self.error = []

        # Initialize axons
        max_axon_len = 8
        self.A = np.zeros(shape=(max_axon_len, size, size))

        # Initialize axons lengths
        self.A_lens = np.random.randint(low=0,
                                        high=max_axon_len,
                                        size=(size, size))

        # Initialize recency array
        self.R = np.zeros(shape=(size,))

        # Intialize activation & weight matrices
        self.x = np.random.random(size=(1, size,)) * 2 - 1
        self.W = np.random.random(size=(size, size)) * 2 - 1

        # Drop out some weights
        indices = np.random.choice(np.arange(self.W.size), replace=False,
                                   size=int(self.W.size * dropout))
        self.W[np.unravel_index(indices, self.W.shape)] = 0
        self.W = self.W * weight_scaling

        # TODO: Topology?

        # Normalize weights
        # rho_W = max(abs(linalg.eig(self.W)[0]))
        # self.W = (1 / rho_W) * self.W
        # self.W = self.W * weight_scaling
        np.fill_diagonal(self.W, 0.)
        self.W[:, -self.out_size:] = 0  # From output, nowhere
        self.W[:self.inp_size, :] = 0  # Nowhere to input

        self.show_plots = show_plots
        if self.show_plots:
            self.fig, self.axes = plt.subplots(2, 5)
            self.ax = self.axes.ravel()
            plt.ion()
            plt.show()

    def plot(self):
        # Plot only at intervals
        if self.time % self.plot_res:
            return
        # Remove axes
        for i in range(len(self.ax)):
            self.ax[i].clear()

        # Transform array into square (for weight heatmap)
        def squarify(arr):
            a = arr.flatten().shape[0]
            b = int(np.ceil(np.sqrt(a)))
            arr = np.concatenate((arr.flatten(), np.zeros(b*b-a)))
            arr = np.reshape(arr, (b, b))
            c = b
            while (b*c-a) >= b:
                c -= 1
                arr = arr[:-1, :]
            return arr

        h = np.asarray(self.x[-self.plot_reach:]).T
        for idx, xline in enumerate(h / self.x[-1].size):
            self.ax[0].plot((xline+idx/self.x[-1].size))
        self.ax[0].set_title("Activations")

        self.ax[1].imshow(squarify(self.W),
                          cmap='coolwarm',
                          vmin=-1, vmax=1,
                          interpolation='nearest')
        self.ax[1].set_title("W")
        self.ax[1].axis('tight')

        self.ax[2].plot(self.x[-self.plot_reach:, -self.out_size:])
        self.ax[2].set_title("Output")
        self.ax[2].plot([self.out(n) for n in range(self.time+2)][-self.plot_reach:])
        self.ax[2].set_title("Target")
        self.ax[2].set_ylim(-1, 1)


        self.ax[3].imshow(squarify(self.R),
                          cmap='gray',
                          vmin=0, vmax=1,
                          interpolation='nearest')
        self.ax[3].set_title("Recency")
        self.ax[3].axis('tight')

        self.ax[4].imshow(squarify(self.receiving),
                          cmap='coolwarm',
                          vmin=-1, vmax=1,
                          interpolation='nearest')
        self.ax[4].set_title("Receiving")
        self.ax[4].axis('tight')

        self.ax[5].imshow(squarify(self.update),
                          cmap='coolwarm',
                          vmin=-1, vmax=1,
                          interpolation='nearest')
        self.ax[5].set_title("Update")
        self.ax[5].axis('tight')

        self.ax[9].imshow(squarify(self.L),
                          cmap='inferno',
                          vmin=0, vmax=1,
                          interpolation='nearest')
        self.ax[9].set_title("Local plasticity")
        self.ax[9].axis('tight')

        self.ax[8].plot(
            np.abs(self.x[-self.plot_reach:, -self.out_size:]
                   - [self.out(n) for n in
                      range(self.time+2)][-self.plot_reach:]))
        self.ax[8].set_title("Error")
        if self.inp(0):
            self.ax[6].plot([self.inp(n) for n in range(1, self.time+1)][-self.plot_reach:])
            self.ax[6].set_title("Input")

        ev = self.A
        ev = np.reshape(ev, (ev.shape[0], ev.shape[1]*ev.shape[2]), order='F').T
        for t in range(ev.shape[1]):
            ev[:, t] *= t + 1
        self.ax[7].eventplot(ev)
        self.ax[7].set_xlim(0, ev.shape[1])
        self.ax[7].set_title("Spike train")

        plt.draw()
        # plt.savefig('plot.pdf')
        plt.pause(0.001)

    def proceed(self):

        # Force input
        self.x[-1][self.x[-1] < 0] *= self.leakage_up
        self.x[-1][self.x[-1] > 0] *= self.leakage_down

        self.x[-1, :self.inp_size] = inp(self.time)

        # Find out which units fire, and insert it in back of dendrites
        if self.spiking:
            firing_units = (self.x[-1] >= self.threshold).astype(int)
        else:
            firing_units = self.x[-1]
        firing_units_rep = np.repeat(
            np.expand_dims(firing_units, axis=0), self.x[-1].size, axis=0)
        self.A = insert_2d_in_3d(self.A, self.A_lens, val=firing_units_rep)

        # Reset firing units except input and output
        self.x[-1, self.inp_size:-self.out_size] = np.where(
            firing_units[self.inp_size:-self.out_size],
            self.reset,
            self.x[-1][self.inp_size:-self.out_size])

        self.R = np.where(firing_units, 1, self.R / np.e)
        self.R[self.R < 1e-3] = 0

        # TODO: Update recency from 2d to 3d array (through time).
        """
        When a neuron fires, use the axon length to find out when its
        postsynaptic neurons are reached. Compute the STDP *now*, and delay
        it in a delay update 3d matrix.

        """

        # Update activations
        self.receiving = self.A[0, :, :] * self.W
        self.x[-1] = f(self.x[-1]
                       + np.dot(self.receiving, self.x[-1])
                       + ((np.random.random(size=self.x[-1].shape) * 2 - 1)
                          * self.noise_factor_a))

        # Evolve axons
        self.A = self.A[1:, :, :]
        self.A = np.pad(self.A, pad_width=((0, 1), (0, 0), (0, 0)))

        # R-STDP
        error = np.abs(self.x[-1, -self.out_size:] - self.out(self.time)) ** 2
        if self.spiking:
            r_rep = np.repeat(np.expand_dims(self.R, axis=0),
                              self.R.size, axis=0)
            Z = r_rep.T - (r_rep / np.e ** self.A_lens)  # PostR - preR [-1, 1]
        else:
            """
            for all presynaptics P:
                update N by G*N*P[-L] if T >(=) L

            """
        presyns = self.x[-np.where(self.A_lens <= self.time, self.A_lens, 0)]

        presyns = np.repeat(
            np.expand_dims(presyns, axis=0), presyns.size, axis=0)
        postsyns = np.repeat(
            np.expand_dims(self.x[-1], axis=0), self.x[-1].size, axis=0).T
        print(postsyns * presyns)
        exit()

        update = 1 - Z * self.lr * error  # []

        # Scale sub-1 to [.5, 1)
        update[update < 1] = update[update < 1] / 2 + 0.5


        if np.max(Z) > 1 or np.min(Z) < -1:
            exit()

        # Metaplasticity
        self.P = ((self.P_current_weight * (1 - np.mean(update ** 2)) + self.P)
                  / (self.P_current_weight + 1))

        # Local plasticity
        self.L = 1 - ((self.L_current_weight * (1 - np.abs(update)) + self.L)
                      / (self.L_current_weight + 1))

        update *= self.L

        W_zeros = np.where(self.W == 0)
        self.W = (self.W
                  * update
                  + (np.random.random(size=self.W.shape) * 2 - 1)
                  * self.noise_factor_w)
        # self.W = np.clip(self.W, a_min=-1, a_max=1)

        # Synaptic scaling
        # m = np.sum(self.W) / np.asarray(np.nonzero(self.W)).size
        # self.W = (self.W - m)
        self.W[W_zeros] = 0
        self.receiving[W_zeros] = 0
        self.L[W_zeros] = 0
        self.update = update
        self.update[W_zeros] = 0

        self.x = np.append(self.x, np.expand_dims(self.x[-1], axis=0), axis=0)
        if self.show_plots:
            self.plot()

        self.time += 1

    def proceed_multiple(self, times):
        while self.time != times:
            self.proceed()

    def mean_error(self):
        errs = np.abs(self.x[:, -self.out_size:]
                   - [self.out(n) for n in range(self.time+1)])
        return sum(errs) / len(errs)


esn = ESN(inp=inp,
          size=12,
          out=out,
          topo='full',
          dropout=0,
          weight_scaling=1,
          show_plots=True)
esn.proceed_multiple(times=-1)
print(esn.mean_error())


"""
Last note:
apparently weakening the weights doesn't mean decreasing them;
it means multiplying them by a sub-one number.
This was the reason the weights from the input were all negative:
because I subtracted the update, rather than factored it.
Now I'm working on rewriting the update step.

"""
