import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def inp(n):
    return np.asarray([])


def out(n, inp):
    return np.asarray([1. * np.sin(n * 0.3)])
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
    def __init__(self, N, dropout, alpha, K, L, show_plots=False):
        np.random.seed(1)
        self.time = 0
        self.plot_res = 1
        self.plot_reach = 140
        self.inp = inp
        self.out = out
        self.K = K
        self.N = N
        self.L = L

        self.weight_scaling = alpha
        self.noise_factor = 0.01
        self.lr = 1
        self.leakage = 0.9

        self.M = []
        self.D = []
        self.P = 0
        self.err = []

        max_axon_len = 12
        self.A = np.zeros(shape=(max_axon_len, N, N))
        stdp_window = 6
        self.F = np.zeros(shape=(stdp_window, N,))
        self.x = np.random.random(size=(N,)) * 2 - 1
        self.A_lens = np.random.randint(low=2,
                                        high=max_axon_len,
                                        size=(N, N))
        self.W = np.random.random(size=(N, N)) * 2 - 1

        indices = np.random.choice(np.arange(self.W.size), replace=False,
                                   size=int(self.W.size * dropout))
        self.W[np.unravel_index(indices, self.W.shape)] = 0

        rho_W = max(abs(linalg.eig(self.W)[0]))
        self.W = (1 / rho_W) * self.W
        self.W = self.W * self.weight_scaling

        self.W_in = np.random.random(size=(N, K)) * 2 - 1
        self.W_out = np.random.random(size=(L, N)) * 2 - 1
        self.W_back = np.random.random(size=(N, L)) * 2 - 1
        self.y = np.random.random(size=(L,)) * 2 - 1

        self.show_plots = show_plots
        if self.show_plots:
            self.fig, self.axes = plt.subplots(2, 4)
            self.ax = self.axes.ravel()
            plt.ion()
            plt.show()

    def plot(self):
        if self.time % self.plot_res:
        # if self.time % int(self.time ** (1/np.e)):
            return
        for i in range(len(self.ax)):
            self.ax[i].clear()

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

        self.ax[0].imshow(squarify(self.x),
                          cmap='coolwarm',
                          vmin=-1, vmax=1,
                          interpolation='nearest')
        self.ax[0].set_title("x")
        self.ax[0].axis('tight')
        if self.W_out is not None:
            self.ax[1].imshow(squarify(self.W_out),
                              cmap='coolwarm',
                              vmin=-1, vmax=1,
                              interpolation='nearest')
            self.ax[1].set_title("W_out")
            self.ax[1].axis('tight')
        self.ax[2].plot([y for (y, _) in self.D][-self.plot_reach:])
        self.ax[2].set_title("y")
        self.ax[2].plot([t for (_, t) in self.D][-self.plot_reach:])
        self.ax[2].set_title("t")
        self.ax[2].set_ylim(-1, 1)

        self.ax[3].imshow(squarify(self.W),
                          cmap='coolwarm',
                          vmin=-1, vmax=1,
                          interpolation='nearest')
        self.ax[3].set_title("W_out")
        self.ax[3].axis('tight')

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
            # self.err)
            moving_average(a=self.err,
                           n=int(1 + self.time // 2)))
        self.ax[5].set_title("err")
        self.ax[5].set_ylim(0, 4)
        plt.draw()
        plt.savefig('plot.png')
        plt.pause(0.001)

    def proceed(self):
        # print(f"t={self.time}")
        # Update activations
        # print(self.W)
        self.A = self.A[1:, :, :]
        self.A = np.pad(self.A, pad_width=((0, 1), (0, 0), (0, 0)))
        self.Fnext = self.F[1:, :]
        self.Fnext = np.pad(self.Fnext, pad_width=((0, 1), (0, 0)))

        self.firing = self.A[0, :, :] * self.W

        next_x = f(np.dot(self.W_in,
                          self.inp(self.time + 1))
                   + np.dot(self.W * self.firing,
                            self.x * self.leakage)
                   + np.dot(self.W_back,
                            self.out(self.time,
                                     self.inp(self.time)))
                   + ((np.random.random(size=self.x.shape) * 2 - 1)
                      * self.noise_factor))
        next_x = np.clip(next_x, a_min=0, a_max = 1)

        self.y = f(np.dot(self.W_out,
                          np.concatenate((self.inp(self.time + 1),
                                          next_x))))

        # Find out which units fire, and insert it in back of dendrites
        firing_units = next_x
        firing_units_rep = np.repeat(
            np.expand_dims(firing_units, axis=0), firing_units.size, axis=0)
        self.Fnext[-1, :] = firing_units
        self.A = insert_2d_in_3d(self.A, self.A_lens, val=firing_units)
        # next_x = np.where(next_x >= self.threshold, self.reset, next_x)

        # R-STDP
        error = (self.y - self.out(self.time, self.inp(self.time)))
        self.err.append(error)

        # print("F", self.F)
        # print("fnext", self.Fnext)

        # print(self.F)
        update = np.zeros(shape=(self.N, self.N))
        def Mdouter(a1, a2):
            print(a1.shape)
            ret = []
            for r in range(a1.shape[0]):
                ret.append(douter(a1[r, :], a2[r, :]))
            return np.asarray(ret)

        def douter(t0, t1):
            return np.outer(t0, t1) - np.outer(t1, t0)  # maybe transpose?

        update = Mdouter(self.F, self.Fnext)

        self.P = ((1 - np.mean(np.abs(update))) + 9*self.P) / 10
        print(self.P)
        logs = np.expand_dims(1 / (1 + np.arange(update.shape[0])), axis=0)
        update = (update.T * logs).T
        update = np.sum(update, axis=0) * (4-error) * self.lr * self.P

        # print(update)

        # print(update)
        self.W += ((np.random.random(size=self.W.shape) * 2 - 1)
                    * self.noise_factor)
        self.W = np.clip(f(self.W + update), a_min=-1, a_max=1)

        self.W[self.W > .9999] = 1
        self.W[self.W < -.9999] = -1

        self.D.append((self.y, self.out(self.time, self.inp(self.time))))

        self.time += 1
        self.x = next_x
        # print(self.x)
        self.F = self.Fnext
        self.update = update
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


esn = ESN(K=0, N=100, L=1, dropout=0.98, alpha=0.5, show_plots=True)
esn.proceed_multiple(times=-1)
print(esn.error())


# def aux(x):
#     x[0] = max(x[0], 1)
#     x[1] = np.clip(x[1], 0, 0.999)
#     x[2] = np.clip(x[2], 0.01, 0.99)
#     x[3] = max(x[0], 32)
#     x[4] = max(x[0], 32)
#     esn = ESN(K=1, N=int(x[0]), L=1, dropout=min(.999, x[1]), alpha=x[2])
#     esn.proceed_multiple(times=int(x[3]))
#     esn.set_training(False)
#     esn.proceed_multiple(times=int(x[4]))
#     return esn.error()

# x0 = [100, .9, .5, 100, 100]
# print(minimize(fun=aux, x0=x0, method='Nelder-Mead'))

