import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def inp(n):
    return np.asarray([])


def out(n, inp):
    return np.asarray([0.2 * np.sin(n/100)])
    # return np.asarray([.42])


def f(x):
    return np.tanh(x)


class ESN():
    def __init__(self, N, dropout, alpha, K, L, show_plots=False):
        self.time = 0
        self.plot_res = 10
        self.inp = inp
        self.out = out
        self.K = K
        self.N = N
        self.L = L
        self.training = True

        self.weight_scaling = alpha
        self.noise_factor = 0.002
        self.leakage = 0.998

        self.M = []
        self.T = []
        self.D = []
        self.init_washout = 2000

        self.x = np.random.random(size=(N,)) * 2 - 1
        self.W = np.random.random(size=(N, N)) * 2 - 1

        indices = np.random.choice(np.arange(self.W.size), replace=False,
                                   size=int(self.W.size * dropout))
        self.W[np.unravel_index(indices, self.W.shape)] = 0

        rho_W = max(abs(linalg.eig(self.W)[0]))
        self.W = (1/rho_W) * self.W
        self.W = self.W * self.weight_scaling

        self.W_in = np.random.random(size=(N, K)) * 2 - 1
        self.W_out = None
        self.W_back = np.random.random(size=(N, L)) * 2 - 1
        self.y = np.random.random(size=(L,)) * 2 - 1

        self.show_plots = show_plots
        if self.show_plots:
            self.fig, self.axes = plt.subplots(2, 3)
            self.ax = self.axes.ravel()
            plt.ion()
            plt.show()

    def plot(self):
        # if self.time % self.plot_res:
        if self.training or self.time % int(self.time ** (1/np.e)):
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
        self.ax[2].plot([y for (y, _) in self.D])
        self.ax[2].set_title("y")
        self.ax[2].plot([t for (_, t) in self.D])
        self.ax[2].set_title("t")

        self.ax[2].plot([(t-y)**2 for (y, t) in self.D])
        self.ax[2].set_title("t")

        self.ax[3].imshow(squarify(self.W),
                          cmap='coolwarm',
                          vmin=-1, vmax=1,
                          interpolation='nearest')
        self.ax[3].set_title("W_out")
        self.ax[3].axis('tight')
        plt.draw()
        plt.savefig('plot.png')
        plt.pause(0.001)

    def proceed(self):

        # Update activations
        self.x = self.x * self.leakage
        next_x = f(np.dot(self.W_in, self.inp(self.time + 1))
                   + np.dot(self.W, self.x)
                   + np.dot(self.W_back, self.out(self.time, self.inp(self.time)))
                   + (np.random.random(size=self.x.shape) * 2 - 1) * self.noise_factor)
        if not self.training:
            self.y = f(np.dot(self.W_out,
                              np.concatenate((self.inp(self.time + 1),
                                              next_x))))
        else:  # Training, so teacher forcing
            self.y = self.out(self.time, self.inp(self.time))

        # Logging
        if self.time >= self.init_washout and self.training:
            self.M.append(np.concatenate((self.inp(self.time), self.x)))
            self.T.append(self.out(self.time, self.inp(self.time)))

        self.D.append((self.y, self.out(self.time, self.inp(self.time))))

        self.time += 1
        self.x = next_x
        if self.show_plots:
            self.plot()

    def proceed_multiple(self, times):
        for n in range(times):
            self.proceed()

    def set_training(self, training):
        self.training = training
        if not training:
            M = np.asarray(self.M)
            T = np.asarray(self.T)
            self.W_out = np.dot(linalg.pinv(M), T).T

    def error(self):
        errs = [(y-t) ** 2 for (y, t) in self.D]
        return sum(errs) / len(errs)

    def __repr__(self):
        return f"{self.time}: {self.x}"


esn = ESN(K=0, N=20, L=1, dropout=.8, alpha=0.2, show_plots=False)
esn.proceed_multiple(times=4000)
esn.set_training(False)
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

