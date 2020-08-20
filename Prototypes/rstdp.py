import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def inp(n):
    return np.asarray([])


def out(n, inp):
    # return np.asarray([1. * np.sin(n * 0.08)])
    return np.asarray([.42])


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
        self.time = 0
        self.plot_res = 20
        self.plot_reach = 140
        self.inp = inp
        self.out = out
        self.K = K
        self.N = N
        self.L = L

        self.weight_scaling = alpha
        self.noise_factor = 0.001
        self.threshold = 0.1
        self.reset = -0.2
        self.lr = 1
        self.leakage = 1

        self.M = []
        self.T = []
        self.D = []
        self.err = []

        max_axon_len = 12
        self.x = np.random.random(size=(N,)) * 2 - 1
        self.A = np.zeros(shape=(max_axon_len, N, N))
        self.A_lens = np.random.randint(low=max_axon_len-3,
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

        self.ax[3].imshow(squarify(self.W),
                          cmap='coolwarm',
                          vmin=-1, vmax=1,
                          interpolation='nearest')
        self.ax[3].set_title("W_out")
        self.ax[3].axis('tight')

        self.ax[4].imshow(squarify(self.firing),
                          cmap='gray',
                          vmin=np.min(self.W), vmax=np.max(self.W),
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

        # Update activations
        self.A = self.A[1:, :, :]
        self.A = np.pad(self.A, pad_width=((0, 1), (0, 0), (0, 0)))

        self.firing = self.A[1, :, :] * self.W

        next_x = f(np.dot(self.W_in,
                          self.inp(self.time + 1))
                   + np.dot(self.firing,
                            self.x * self.leakage)
                   + np.dot(self.W_back,
                            self.out(self.time,
                                     self.inp(self.time)))
                   + ((np.random.random(size=self.x.shape) * 2 - 1)
                      * self.noise_factor))

        self.y = f(np.dot(self.W_out,
                          np.concatenate((self.inp(self.time + 1),
                                          next_x))))
        # Find out which units fire, and insert it in back of dendrites
        x_rep = np.repeat(np.expand_dims(next_x, axis=0), next_x.size, axis=0)
        firing_units = x_rep >= self.threshold
        self.A = insert_2d_in_3d(self.A, self.A_lens, val=firing_units)
        next_x = np.where(next_x >= self.threshold, self.reset, next_x)

        # R-STDP
        error = (self.y - self.out(self.time, self.inp(self.time))) ** 2
        self.err.append(error)
        # print("error", error)
        almost_fire = (self.A[2, :, :] != 0)
        just_fired = (self.A[1, :, :] != 0)
        rec_act = np.repeat(np.expand_dims(next_x, axis=0), self.N, axis=0).T
        fired_factor = almost_fire + just_fired * -1
        # print("error", error)
        # print("fired_factor", np.min(fired_factor), np.max(fired_factor))
        # print("rec_act", rec_act)
        update = 1 - error * self.lr * fired_factor * np.abs(rec_act)
        # print("update", update)
        # print("x", self.x)
        # print("W", self.W)
        self.update = update
        # TODO: also include the absolute activation of the receiving neuron.
        # TODO: also increase the weight strength (note: ceil!)
        # * either: if the weight fired this(/previous) round and the receiver is now large absolute
        # * either: by a small constant every time (hacky?)

        # print(update)
        self.W += ((np.random.random(size=self.W.shape) * 2 - 1)
                    * self.noise_factor)
        self.W = np.clip(self.W * update, a_min=-1, a_max=1)

        self.W[self.W > .9999] = 1
        self.W[self.W < -.9999] = -1

        self.D.append((self.y, self.out(self.time, self.inp(self.time))))

        self.time += 1
        self.x = next_x
        if self.show_plots:
            self.plot()

    def proceed_multiple(self, times):
        for n in range(times):
            self.proceed()

    def error(self):
        errs = [(y-t) ** 2 for (y, t) in self.D]
        return sum(errs) / len(errs)

    def __repr__(self):
        return f"{self.time}: {self.x}"


esn = ESN(K=0, N=8, L=1, dropout=0, alpha=0.7, show_plots=True)
esn.proceed_multiple(times=2000)
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

