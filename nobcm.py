import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def inp(n):
    return np.asarray([])


def out(n):
    return np.asarray([1. * np.sin(n * 0.03)])
    # return np.asarray([.42])


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
        np.random.seed(1)
        self.time = 0
        self.plot_res = 20
        self.plot_reach = 140
        self.inp = inp
        self.inp_size = inp(0).size
        self.out = out
        self.out_size = out(0).size
        self.size = size

        self.weight_scaling = weight_scaling
        self.noise_factor_w = 0.
        self.noise_factor_a = 0.
        self.lr = 1
        self.leakage = 1.001
        self.hist_x = []
        self.M = []
        self.D = []
        self.P = 1
        self.err = []

        # Initialize axons
        max_axon_len = 16
        self.A = np.zeros(shape=(max_axon_len, size, size))

        # Initialize axons lengths
        self.A_lens = np.random.randint(low=2,
                                        high=max_axon_len,
                                        size=(size, size))

        # Initialize STDP window
        stdp_window = 6
        self.F = np.zeros(shape=(stdp_window, size,))

        # Intialize activation & weight matrices
        self.x = np.random.random(size=(size,)) * 2 - 1
        self.W = np.random.random(size=(size, size)) * 2 - 1

        # Drop out some weights
        indices = np.random.choice(np.arange(self.W.size), replace=False,
                                   size=int(self.W.size * dropout))
        self.W[np.unravel_index(indices, self.W.shape)] = 0

        # TODO: Drop out weights to input and from output?

        # Normalize weights
        rho_W = max(abs(linalg.eig(self.W)[0]))
        self.W = (1 / rho_W) * self.W
        self.W = self.W * self.weight_scaling

        self.show_plots = show_plots
        if self.show_plots:
            self.fig, self.axes = plt.subplots(2, 4)
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
        h = np.asarray(self.hist_x[-self.plot_reach:]).T
        for idx, xline in enumerate(h / self.x.size):
            self.ax[0].plot((xline+idx/self.x.size))
        self.ax[0].set_title("Activations")

        self.ax[1].imshow(squarify(self.W),
                          cmap='coolwarm',
                          vmin=-1, vmax=1,
                          interpolation='nearest')
        self.ax[1].set_title("W")
        self.ax[1].axis('tight')

        self.ax[2].plot([y for (y, _) in self.D][-self.plot_reach:])
        self.ax[2].set_title("Output")
        self.ax[2].plot([t for (_, t) in self.D][-self.plot_reach:])
        self.ax[2].set_title("Target")
        self.ax[2].set_ylim(-1, 1)


        self.ax[4].imshow(squarify(self.firing),
                          cmap='gray',
                          vmin=0, vmax=1,
                          interpolation='nearest')
        self.ax[4].set_title("Firing")
        self.ax[4].axis('tight')

        self.ax[6].imshow(squarify(self.update),
                          cmap='gray',
                          vmin=0, vmax=2,
                          interpolation='nearest')
        self.ax[6].set_title("Update")
        self.ax[6].axis('tight')

        self.ax[5].plot(
            self.err)
            # moving_average(a=self.err,
            #                n=int(1 + self.time // 2)))
        self.ax[5].set_title("Error")
        plt.draw()
        plt.savefig('plot.pdf')
        plt.pause(0.001)

    def proceed(self):
        # Update activations
        self.x[:self.inp_size] = inp(self.time)
        
        # Evolve axons
        self.A = self.A[1:, :, :]
        self.A = np.pad(self.A, pad_width=((0, 1), (0, 0), (0, 0)))

        self.firing = self.A[0, :, :] * self.W

        # Update activations
        self.x = f(self.x * self.leakage
                   + np.dot(self.firing, self.x)
                   + ((np.random.random(size=self.x.shape) * 2 - 1)
                      * self.noise_factor_a))
        # next_x = np.clip(next_x, a_min=0, a_max = 1)

        # Find out which units fire, and insert it in back of dendrites
        firing_units_rep = np.repeat(
            np.expand_dims(self.x, axis=0), self.x.size, axis=0)
        self.A = insert_2d_in_3d(self.A, self.A_lens, val=self.x)

        # Update STDP window
        old_F = self.F
        self.F = self.F[1:, :]
        self.F = np.pad(self.F, pad_width=((0, 1), (0, 0)))
        self.F[-1, :] = self.x
        

        # R-STDP
        error = np.abs(self.x[-self.out_size:] - self.out(self.time))
        self.err.append(error)

        update = np.zeros(shape=(self.size, self.size))
        def Mdouter(a1, a2):
            ret = []
            for r in range(a1.shape[0]):
                ret.append(douter(a1[r, :], a2[r, :]))
            return np.asarray(ret)

        def douter(t0, t1):
            return np.outer(t0, t1) - np.outer(t1, t0)  # maybe transpose?

        update = Mdouter(old_F, self.F)

        # Metaplasticity
        self.P = ((1 - np.mean(update ** 2)) + 1*self.P) / 2
        print(self.P)

        logs = np.expand_dims(1 / (1 + np.arange(update.shape[0])), axis=0)
        update = (update.T * logs).T
        update = np.sum(update, axis=0) * -error * self.lr * self.P

        self.W += update + ((np.random.random(size=self.W.shape) * 2 - 1)
                            * self.noise_factor_w)

        # Synaptic scaling
        m = np.sum(self.W) / np.asarray(np.nonzero(self.W)).size
        self.W = f(self.W-m)

        # self.W[self.W > .9999] = 1
        # self.W[self.W < -.9999] = -1

        self.D.append((self.x[-self.out_size:], self.out(self.time)))

        self.time += 1

        self.update = update
        self.hist_x.append(self.x)
        if self.show_plots:
            self.plot()

    def proceed_multiple(self, times):
        while self.time != times:
            self.proceed()

    def error(self):
        errs = [(y-t) for (y, t) in self.D]
        return sum(errs) / len(errs)

    def __repr__(self):
        return f"{self.time}: {self.x}"


esn = ESN(inp=inp, 
          size=20, 
          out=out,
          topo='full',
          dropout=0.95, 
          weight_scaling=0.9, 
          show_plots=True)
esn.proceed_multiple(times=-1)
print(esn.error())