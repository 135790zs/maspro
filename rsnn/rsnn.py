from config import config
import numpy as np
import utils2 as uts
import matplotlib.pyplot as plt


def traub_izh_unconn():
    fig, axs = plt.subplots(8, 1)

    log_v1 = []
    log_v2 = []
    log_evec_v1 = []
    log_evec_u1 = []
    log_evec_v2 = []
    log_evec_u2 = []
    log_h2 = []
    log_etrace1 = []
    log_grad1 = []
    log_etrace2 = []
    log_grad2 = []
    log_u1 = []
    log_u2 = []
    log_z1 = []
    log_z2 = []

    num = 2
    Nv = np.ones(shape=(num,)) * config["eqb"]
    Nu = np.zeros(shape=(num,))
    Nz = np.zeros(shape=(num,))
    H = np.zeros(shape=(num,))
    TZ = np.zeros(shape=(num,))
    I = np.zeros(shape=(num,))

    rng = np.random.default_rng()

    W = rng.random(size=(num, num,))
    EVv = np.zeros(shape=W.shape)
    EVu = np.zeros(shape=W.shape)
    ET = np.zeros(shape=W.shape)
    G = np.zeros(shape=W.shape)

    T = 5000
    A = 15/2
    B = 5/2
    C = 10

    for t in range(0, T):
        if t < T * 0.5:
            I[0] = A
            I[1] = B
            if TZ[0] >= TZ[1]:
                I[1] = C
        else:
            I[0] = B
            I[1] = A
            if TZ[1] >= TZ[0]:
                I[0] = C

        Nvp = uts.V_next(Nv=Nv, Nu=Nu, Nz=Nz, I=I)

        Nz = np.where(Nvp >= config["thr"], 1., 0.)
        TZ = np.where(Nvp >= config["thr"], t, TZ)

        # Should this operate on pseudo?
        Nun = uts.U_next(Nu=Nu, Nz=Nz, Nv=Nvp)
        Nvn = uts.V_next(Nu=Nu, Nz=Nz, Nv=Nvp, I=I)

        EVvn = uts.EVv_next(EVv=EVv, EVu=EVu, Nz=Nz, Nv=Nvp)
        EVun = uts.EVu_next(EVv=EVv, EVu=EVu, Nz=Nz)

        EVv = EVvn
        EVu = EVun

        Nv = Nvn
        Nu = Nun

        H = uts.H_next(Nv=Nv)
        ET = H * EVv

        G = G + ET

        log_v1.append(Nv[0])
        log_v2.append(Nv[1])
        log_evec_v1.append(EVv[0, 1])
        log_evec_v2.append(EVv[1, 0])
        log_evec_u1.append(EVu[0, 1])
        log_evec_u2.append(EVu[1, 0])
        log_etrace1.append(ET[0, 1])
        log_etrace2.append(ET[1, 0])
        log_grad1.append(G[0, 1])
        log_grad2.append(G[1, 0])
        log_h2.append(H[1])
        log_u1.append(Nu[0])
        log_u2.append(Nu[1])
        log_z1.append(Nz[0])
        log_z2.append(Nz[1])

    axs[0].plot(log_v1)
    axs[0].plot(log_v2)
    axs[0].set_title("Voltage")
    axs[1].plot(log_u1)
    axs[1].plot(log_u2)
    axs[1].set_title("Refractory")
    axs[2].plot(log_z1)
    axs[2].plot(log_z2)
    axs[2].set_title("Spikes")
    axs[3].plot(log_evec_v1)
    axs[3].plot(log_evec_v2)
    axs[3].set_title("Voltage eligibility")
    axs[4].plot(log_evec_u1)
    axs[4].plot(log_evec_u2)
    axs[4].set_title("Refractory eligibility")
    axs[5].plot(log_h2)
    axs[5].set_title("Pseudo-derivative")
    axs[6].plot(log_etrace1)
    axs[6].plot(log_etrace2)
    axs[6].set_title("Eligibility trace")
    axs[7].plot(log_grad1)
    axs[7].plot(log_grad2)
    axs[7].set_title("Gradient")

    plt.show()


def traub_izh_conn():
    fig, axs = plt.subplots(8, 1)

    log_v1 = []
    log_v2 = []
    log_evec_v1 = []
    log_evec_u1 = []
    log_evec_v2 = []
    log_evec_u2 = []
    log_h2 = []
    log_etrace1 = []
    log_W1 = []
    log_etrace2 = []
    log_W2 = []
    log_u1 = []
    log_u2 = []
    log_z1 = []
    log_z2 = []

    num = 2
    Nv = np.ones(shape=(num,)) * config["eqb"]
    Nu = np.zeros(shape=(num,))
    Nz = np.zeros(shape=(num,))
    H = np.zeros(shape=(num,))
    TZ = np.zeros(shape=(num,))
    I = np.zeros(shape=(num,))

    rng = np.random.default_rng()

    W = rng.random(size=(num, num,))
    np.fill_diagonal(W, 0.)

    EVv = np.zeros(shape=W.shape)
    EVu = np.zeros(shape=W.shape)
    ET = np.zeros(shape=W.shape)

    T = 5000
    A = 15/2
    B = 5/2
    C = 10

    for t in range(0, T):
        I = np.dot(W, Nz)

        # Mock input
        if t % 2 == 0 and t < 0.5 * T:
            I[0] += 10
        if t % 3 == 1 and t < 0.5 * T:
            I[1] += 24

        print(W)

        Nvp = uts.V_next(Nv=Nv, Nu=Nu, Nz=Nz, I=I)

        Nz = np.where(Nvp >= config["thr"], 1., 0.)
        TZ = np.where(Nvp >= config["thr"], t, TZ)

        # Should this operate on pseudo?
        Nun = uts.U_next(Nu=Nu, Nz=Nz, Nv=Nvp)
        Nvn = uts.V_next(Nu=Nu, Nz=Nz, Nv=Nvp, I=I)

        EVvn = uts.EVv_next(EVv=EVv, EVu=EVu, Nz=Nz, Nv=Nvp)
        EVun = uts.EVu_next(EVv=EVv, EVu=EVu, Nz=Nz)

        EVv = EVvn
        EVu = EVun

        Nv = Nvn
        Nu = Nun

        H = uts.H_next(Nv=Nv)
        ET = H * EVv

        W = W + ET
        np.fill_diagonal(W, 0)

        log_v1.append(Nv[0])
        log_v2.append(Nv[1])
        log_evec_v1.append(EVv[0, 1])
        log_evec_v2.append(EVv[1, 0])
        log_evec_u1.append(EVu[0, 1])
        log_evec_u2.append(EVu[1, 0])
        log_etrace1.append(ET[0, 1])
        log_etrace2.append(ET[1, 0])
        log_W1.append(W[0, 1])
        log_W2.append(W[1, 0])
        log_h2.append(H[1])
        log_u1.append(Nu[0])
        log_u2.append(Nu[1])
        log_z1.append(Nz[0])
        log_z2.append(Nz[1])

    axs[0].plot(log_v1)
    axs[0].plot(log_v2)
    axs[0].set_title("Voltage")
    axs[1].plot(log_u1)
    axs[1].plot(log_u2)
    axs[1].set_title("Refractory")
    axs[2].plot(log_z1)
    axs[2].plot(log_z2)
    axs[2].set_title("Spikes")
    axs[3].plot(log_evec_v1)
    axs[3].plot(log_evec_v2)
    axs[3].set_title("Voltage eligibility")
    axs[4].plot(log_evec_u1)
    axs[4].plot(log_evec_u2)
    axs[4].set_title("Refractory eligibility")
    axs[5].plot(log_h2)
    axs[5].set_title("Pseudo-derivative")
    axs[6].plot(log_etrace1)
    axs[6].plot(log_etrace2)
    axs[6].set_title("Eligibility trace")
    axs[7].plot(log_W1)
    axs[7].plot(log_W2)
    axs[7].set_title("Gradient")

    plt.show()

traub_izh_conn()


# Next time: Make a single-layer DRSNN
