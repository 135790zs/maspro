from config import cfg
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
import sys
from matplotlib import rcParams as rc

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'
np.set_printoptions(threshold=sys.maxsize)


def traub_lif(T=1000, num=2):  # WORKING

    X = ut.get_artificial_input(T=T,
                                num=num,
                                dur=18,
                                diff=5,
                                interval=100,
                                val=3.55,
                                switch_interval=500)

    # Logging arrays
    log = {
        "Nv": np.zeros(shape=(T, num,)),
        "X": X,
        "Nz": np.zeros(shape=(T, num,)),
        "H": np.zeros(shape=(T, num,)),
        "EVv": np.zeros(shape=(T, num, num,)),
        "ET": np.zeros(shape=(T, num, num,)),
        "W": np.zeros(shape=(T, num, num,)),
    }

    # Variable arrays
    Nv = np.ones(shape=(num,))
    Nz = np.zeros(shape=(num,))
    H = np.zeros(shape=(num,))
    TZ = np.zeros(shape=(num,))

    rng = np.random.default_rng()
    W = rng.random(size=(num, num,))
    np.fill_diagonal(W, 0.)

    EVv = np.zeros(shape=(num, num,))
    ET = np.zeros(shape=(num, num,))

    for t in range(0, T):

        I = X[t, :]

        Nz = np.where(np.logical_and(t - TZ > cfg["dt_refr"],
                                     Nv >= cfg["thr"]),
                      1,
                      0)
        TZ = np.where(Nz, t, TZ)

        R = (t - TZ <= cfg["dt_refr"]).astype(int)

        Nv = cfg["alpha"] * Nv + I - Nz * cfg["alpha"] * Nv - R * cfg["alpha"] * Nv

        EVv = cfg["alpha"] * (1 - Nz - R) * EVv + Nz[np.newaxis].T

        H = np.where(t - TZ < cfg["dt_refr"],
                     -cfg["gamma"],
                     cfg["gamma"] * np.clip(a=1 - (abs(Nv - cfg["thr"])
                                                   / cfg["thr"]),
                                            a_min=0,
                                            a_max=None))

        ET = H * EVv

        W = W + ET

        log["Nv"][t, :] = Nv
        log["Nz"][t, :] = Nz
        log["H"][t, :] = H
        log["EVv"][t, :, :] = EVv
        log["ET"][t, :, :] = ET
        log["W"][t, :, :] = W

    ut.plot_logs(log, title="STDP-LIF")


def bellec_alif_stdp():  # WORKING
    fig, axs = plt.subplots(8, 1)

    log_v1 = []
    log_v2 = []
    log_u1 = []
    log_u2 = []
    log_evec_v = []
    log_evec_u = []
    log_h2 = []
    log_etrace = []
    log_grad = []
    log_z1 = []
    log_z2 = []

    v1 = 0
    v2 = 0
    u1 = cfg["H1"]
    u2 = cfg["H1"]
    z1 = 0
    z2 = 0
    evv = 0
    evu = 0
    h2 = 0
    et = 0
    grad = 0

    alpha = 0.9
    beta = 0.07
    rho = np.e**(-1/200)
    dt_ref = 3

    T = 250
    tzi = 0
    tzo = 0
    A = 2
    B = 2
    C = 15

    for t in range(0, T):
        if t < T * 0.5:
            I_i = A
            I_o = B
            if tzi >= tzo:
                I_o = C
        else:
            I_i = B
            I_o = A
            if tzo >= tzi:
                I_i = C

        if v1 >= u1 and t - tzi >= dt_ref:
            z1 = 1
            tzi = t
        else:
            z1 = 0

        if v2 >= u2 and t - tzo >= dt_ref:
            z2 = 1
            tzo = t
        else:
            z2 = 0

        v1 = alpha*v1 + I_i - z1*alpha*v1 - int((t-tzi) == dt_ref)*alpha*v1
        v2 = alpha*v2 + I_o - z2*alpha*v2 - int((t-tzo) == dt_ref)*alpha*v2
        u1 = rho * u1 + z1
        u2 = rho * u2 + z2

        h2 = -cfg["gamma"] if t-tzo < dt_ref else cfg["gamma"] * max(0, 1 - abs((v2 - cfg["H1"]) / cfg["H1"]))
        evvn = alpha*(1-z2-int((t-tzo) == dt_ref)) * evv + z1
        evu = h2*evv + (rho - h2*beta)*evu
        evv = evvn
        et = h2 * evv

        grad = grad + et

        log_v1.append(v1)
        log_v2.append(v2)
        log_u1.append(u1)
        log_u2.append(u2)
        log_evec_v.append(evv)
        log_evec_u.append(evu)
        log_etrace.append(et)
        log_grad.append(grad)
        log_h2.append(h2)
        log_z1.append(z1)
        log_z2.append(z2)

    axs[0].plot(log_v1)
    axs[0].plot(log_v2)
    axs[0].set_title("Voltage")
    axs[1].plot(log_u1)
    axs[1].plot(log_u2)
    axs[1].set_title("Adaptive threshold")
    axs[2].plot(log_z1)
    axs[2].plot(log_z2)
    axs[2].set_title("Spikes")
    axs[3].plot(log_evec_v)
    axs[3].set_title("Voltage eligibility")
    axs[4].plot(log_evec_u)
    axs[4].set_title("Threshold eligibility")
    axs[5].plot(log_h2)
    axs[5].set_title("Pseudo-derivative")
    axs[6].plot(log_etrace)
    axs[6].set_title("Eligibility trace")
    axs[7].plot(log_grad)
    axs[7].set_title("Gradient")
    plt.show()


def traub_izh(T=1000, num=2, uses_weights=False):  #

    X = ut.get_artificial_input(T=T,
                                num=num,
                                dur=20,
                                diff=5,
                                interval=100,
                                val=25,
                                switch_interval=500)

    # Logging arrays
    log = {
        "Nv": np.zeros(shape=(T, num,)),
        "X": X,
        "Nu": np.zeros(shape=(T, num,)),
        "Nz": np.zeros(shape=(T, num,)),
        "H": np.zeros(shape=(T, num,)),
        "EVv": np.zeros(shape=(T, num, num,)),
        "EVu": np.zeros(shape=(T, num, num,)),
        "ET": np.zeros(shape=(T, num, num,)),
        "W": np.zeros(shape=(T, num, num,)),
    }

    # Variable arrays
    Nv = np.ones(shape=(num,)) * cfg["eqb"]
    Nu = np.ones(shape=(num,)) * -10
    Nz = np.zeros(shape=(num,))
    H = np.zeros(shape=(num,))
    TZ = np.zeros(shape=(num,))

    rng = np.random.default_rng()
    W = rng.random(size=(num, num,))
    np.fill_diagonal(W, 0.)

    EVv = np.zeros(shape=(num, num,))
    EVu = np.zeros(shape=(num, num,))
    ET = np.zeros(shape=(num, num,))

    for t in range(0, T):
        I = np.dot(W, Nz) if uses_weights else np.zeros(shape=(num,))

        I += X[t, :]

        Nz = np.where(Nv >= cfg["thr"], 1., 0.)
        TZ = np.where(Nv >= cfg["thr"], t, TZ)

        # Should this operate on pseudo?
        Nun = ut.U_next(Nu=Nu, Nz=Nz, Nv=Nv)
        Nvn = ut.V_next(Nu=Nu, Nz=Nz, Nv=Nv, I=I)

        EVvn = ut.EVv_next(EVv=EVv, EVu=EVu, Nz=Nz, Nv=Nv)
        EVun = ut.EVu_next(EVv=EVv, EVu=EVu, Nz=Nz)

        EVv = EVvn
        EVu = EVun

        Nv = Nvn
        Nu = Nun

        H = ut.H_next(Nv=Nv)
        ET = H * EVv

        W = W + ET
        np.fill_diagonal(W, 0)

        log["Nv"][t, :] = Nv
        log["Nu"][t, :] = Nu
        log["Nz"][t, :] = Nz
        log["H"][t, :] = H
        log["EVv"][t, :, :] = EVv
        log["EVu"][t, :, :] = EVu
        log["ET"][t, :, :] = ET
        log["W"][t, :, :] = W

    ut.plot_logs(log)


# traub_lif()
traub_izh()
