from config import config
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

# Traub alg 2

rng = np.random.default_rng()


def U(a, b):
    return a + rng.random() * (b - a)


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

    T = 500
    tzi = 0
    tzo = 0
    A = 15/2
    B = 5/2
    C = 10

    for t in range(0, T):
        if t < T * 0.3:
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

        evvn = ut.evvnext(zi=z1, zj=z2, vi=v1, vj=v2, evv=evv, evu=evu)
        evun = ut.evunext(zi=z1, zj=z2, evv=evv, evu=evu)

        evv = evvn
        evu = evun

        v1 = v1n
        u1 = u1n
        v2 = v2n
        u2 = u2n
        h2 = ut.h(v=v2_pseudo)

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
