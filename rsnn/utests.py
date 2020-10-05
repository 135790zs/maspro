from config import config
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

# Traub alg 2

rng = np.random.default_rng()


def U(a, b):
    return a + rng.random() * (b - a)

def traub_izh_np():
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
    Nv = np.ones(shape=(num,)) * -config["IzhV1"]
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

    T = 500
    A = 5/2
    B = 10/2
    C = 15/2
    D = 10

    for t in range(0, T):
        # if t < T * 0.33:
        #     I[0] = A
        #     I[1] = B
        #     I[2] = C
        #     if TZ[0] >= TZ[1]:
        #         I[1] = D
        # elif t < T * 0.67:
        #     I[0] = B
        #     I[1] = C
        #     I[2] = A
        #     if TZ[1] >= TZ[2]:
        #         I[2] = D

        # else:
        #     I[0] = C
        #     I[1] = A
        #     I[2] = B
        #     if TZ[1] >= TZ[2]:
        #         I[2] = D
        if t < T * 0.5:
            I[0] = C
            I[1] = A
            if TZ[0] >= TZ[1]:
                I[1] = D
        else:
            I[0] = A
            I[1] = C
            if TZ[1] >= TZ[0]:
                I[0] = D

        Nvp = (Nv - (Nv + 65) * Nz) + config["dt"] * (0.04 * (Nv - (Nv + 65) * Nz)**2
                                                      + 5 * (Nv - (Nv + 65) * Nz)
                                                      + 140
                                                      - (Nu + 2 * Nz)
                                                      + I)

        Nz = np.where(Nvp >= config["H1"], 1, 0)
        TZ = np.where(Nvp >= config["H1"], TZ, t)

        # Should this operate on pseudo?
        Nun = (Nu + 2 * Nz) + config["dt"] * (0.004 * Nvp - 0.02 * (Nu + 2 * Nz))
        Nvn = (Nvp - (Nvp + 65) * Nz) + config["dt"] * (0.04 * (Nvp - (Nvp + 65) * Nz)**2
                                                        + 5 * (Nvp - (Nvp + 65) * Nz)
                                                        + 140
                                                        - (Nu + 2 * Nz)
                                                        + I)

        EVvn = (EVv * (1 - Nz
                       + 0.08 * config["dt"] * Nvp
                       - 0.08 * config["dt"] * Nvp * Nz  # not sure about Nvp here
                       + 5 * config["dt"]
                       - 5 * config["dt"] * Nz)
                - EVu * config["dt"]
                + Nz[np.newaxis].T * config["dt"])

        EVun = (EVv * (0.004 * config["dt"]
                       - 0.004 * config["dt"] * Nz)
                + EVu * (1
                         - 0.02 * config["dt"]))

        H = config["gamma"] * np.exp((np.clip(Nvp, a_min=None, a_max=config["H1"]) - config["H1"])
                                     / config["H1"])

        EVv = EVvn
        EVu = EVun

        Nv = Nvn
        Nu = Nun

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

def traub_izh():
    fig, axs = plt.subplots(8, 1)

    log_v1 = []
    log_v2 = []
    log_evec_v = []
    log_evec_u = []
    log_h2 = []
    log_etrace = []
    log_grad = []
    log_u1 = []
    log_u2 = []
    log_z1 = []
    log_z2 = []

    v1 = -config["IzhV1"]
    v2 = -config["IzhV1"]
    u1 = 0
    u2 = 0
    z1 = 0
    z2 = 0
    evv = 0
    evu = 0
    h2 = 0
    et = 0
    grad = 0

    T = 5000
    tzi = 0
    tzo = 0
    A = 15/2
    B = 5/2
    C = 10

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

        v1_pseudo = ut.vnext(v=v1, u=u1, z=z1, I=I_i)
        v2_pseudo = ut.vnext(v=v2, u=u2, z=z2, I=I_o)

        if v1_pseudo >= config["H1"]:
            z1 = 1
            tzi = t
        else:
            z1 = 0

        if v2_pseudo >= config["H1"]:
            z2 = 1
            tzo = t
        else:
            z2 = 0

        v1n = ut.vnext(v=v1_pseudo, u=u1, z=z1, I=I_i)
        u1n = ut.unext(v=v1_pseudo, u=u1, z=z1)
        v2n = ut.vnext(v=v2_pseudo, u=u2, z=z2, I=I_o)
        u2n = ut.unext(v=v2_pseudo, u=u2, z=z2)

        evvn = ut.evvnext(zi=z1, zj=z2, vi=v1_pseudo, vj=v2_pseudo, evv=evv, evu=evu)
        evun = ut.evunext(zi=z1, zj=z2, evv=evv, evu=evu)

        evv = evvn
        evu = evun

        v1 = v1n
        u1 = u1n
        v2 = v2n
        u2 = u2n

        h2 = ut.h(v=v2)
        et = h2 * evv

        grad = grad + et

        log_v1.append(v1)
        log_v2.append(v2)
        log_evec_v.append(evv)
        log_evec_u.append(evu)
        log_etrace.append(et)
        log_grad.append(grad)
        log_h2.append(h2)
        log_u1.append(u1)
        log_u2.append(u2)
        log_z1.append(z1)
        log_z2.append(z2)

    axs[0].plot(log_v1)
    axs[0].plot(log_v2)
    axs[0].set_title("Voltage")
    axs[1].plot(log_u1)
    axs[1].plot(log_u2)
    axs[1].set_title("Refractory")
    axs[2].plot(log_z1)
    axs[2].plot(log_z2)
    axs[2].set_title("Spikes")
    axs[3].plot(log_evec_v)
    axs[3].set_title("Voltage eligibility")
    axs[4].plot(log_evec_u)
    axs[4].set_title("Refractory eligibility")
    axs[5].plot(log_h2)
    axs[5].set_title("Pseudo-derivative")
    axs[6].plot(log_etrace)
    axs[6].set_title("Eligibility trace")
    axs[7].plot(log_grad)
    axs[7].set_title("Gradient")

    plt.show()


def traub_lif():  # WORKING
    fig, axs = plt.subplots(8, 1)

    log_v1 = []
    log_v2 = []
    log_evec = []
    log_h2 = []
    log_etrace = []
    log_grad = []
    log_z1 = []
    log_z2 = []

    v1 = 0
    v2 = 0
    z1 = 0
    z2 = 0
    ev = 0
    h2 = 0
    et = 0
    grad = 0

    alpha = 0.9
    dt_ref = 3

    T = 250
    tzi = 0
    tzo = 0
    A = 5
    B = 5
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

        if v1 >= config["H1"] and t - tzi >= dt_ref:
            z1 = 1
            tzi = t
        else:
            z1 = 0

        if v2 >= config["H1"] and t - tzo >= dt_ref:
            z2 = 1
            tzo = t
        else:
            z2 = 0

        v1 = alpha*v1 + I_i - z1*alpha*v1 - int((t-tzi) == dt_ref)*alpha*v1
        v2 = alpha*v2 + I_o - z2*alpha*v2 - int((t-tzo) == dt_ref)*alpha*v2

        ev = alpha*(1-z2-int((t-tzo) == dt_ref)) * ev + z1
        h2 = -config["gamma"] if t-tzo < dt_ref else config["gamma"] * max(0, 1 - abs((v2 - config["H1"]) / config["H1"]))
        et = h2 * ev

        grad = grad + et

        log_v1.append(v1)
        log_v2.append(v2)
        log_evec.append(ev)
        log_etrace.append(et)
        log_grad.append(grad)
        log_h2.append(h2)
        log_z1.append(z1)
        log_z2.append(z2)

    axs[0].plot(log_v1)
    axs[0].plot(log_v2)
    axs[0].set_title("Voltage")
    axs[2].plot(log_z1)
    axs[2].plot(log_z2)
    axs[2].set_title("Spikes")
    axs[3].plot(log_evec)
    axs[3].set_title("Voltage eligibility")
    axs[5].plot(log_h2)
    axs[5].set_title("Pseudo-derivative")
    axs[6].plot(log_etrace)
    axs[6].set_title("Eligibility trace")
    axs[7].plot(log_grad)
    axs[7].set_title("Gradient")
    plt.show()


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
    u1 = config["H1"]
    u2 = config["H1"]
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

        h2 = -config["gamma"] if t-tzo < dt_ref else config["gamma"] * max(0, 1 - abs((v2 - config["H1"]) / config["H1"]))
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


traub_izh()
