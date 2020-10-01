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

    s1 = np.asarray([-config["IzhV1"], -10.])
    s2 = np.asarray([-config["IzhV1"], -10.])
    z1 = 0
    z2 = 0
    ev = np.zeros(shape=(2,))
    h2 = 0
    et = 0
    grad = 0

    T = 5000
    tzi = 0
    tzo = 0
    A = 15
    B = 5
    # C = 15

    for t in range(0, T):
        if t < T * 0.5:
            I_i = U(1, A)
            I_o = U(1, B)
            if tzi > tzo:
                I_o = U(0, 1)*(t-tzi)
        else:
            I_i = U(1, B)
            I_o = U(1, A)
            if tzo > tzi:
                I_i = U(0, 1)*(t-tzo)

        if s1[0] > config["H1"]:
            z1 = 1
            tzi = t
        else:
            z1 = 0

        if s2[0] > config["H1"]:
            z2 = 1
            tzo = t
        else:
            z2 = 0

        s1vn = ut.vnext(v=s1[0], u=s1[1], z=z1, I=I_i)
        s1un = ut.unext(v=s1[0], u=s1[1], z=z1)
        s2vn = ut.vnext(v=s2[0], u=s2[1], z=z2, I=I_o)
        s2un = ut.unext(v=s2[0], u=s2[1], z=z2)
        evvn = ut.evvnext(zi=z1, zj=z2, vj=s2[0], evv=ev[0], evu=ev[1])
        evun = ut.evunext(zi=z1, zj=z2, evv=ev[0], evu=ev[1])

        s1[0] = s1vn
        s1[1] = s1un
        s2[0] = s2vn
        s2[1] = s2un

        ev[0] = evvn
        ev[1] = evun

        h2 = ut.h(v=s2[0])
        et = h2 * ev[1]

        grad = grad + et

        log_v1.append(s1[0])
        log_v2.append(s2[0])
        log_evec_v.append(ev[0])
        log_evec_u.append(ev[1])
        log_etrace.append(et)
        log_grad.append(grad)
        log_h2.append(h2)
        log_u1.append(s1[1])
        log_u2.append(s2[1])
        log_z1.append(z1)
        log_z2.append(z2)
        # if abs(ev[0]) > 10:
        #     b = 5
        #     print("Log v1", log_v1[-b:])
        #     print("Log v2", log_v2[-b:])
        #     print("EvecV", log_evec_v[-b:])
        #     print("EvecU", log_evec_u[-b:])
        #     exit()

    axs[0].plot(log_v1)
    axs[0].plot(log_v2)
    axs[0].axhline(y=117, c='black', alpha=0.3, linestyle='--', linewidth=1)
    axs[0].axhline(y=-80, c='black', alpha=0.3, linestyle='--', linewidth=1)
    axs[0].set_title("Voltage")
    # axs[0].set_ylim((-100, 130))
    axs[1].plot(log_u1)
    axs[1].plot(log_u2)
    axs[1].set_title("Refractory")
    axs[2].plot(log_z1)
    axs[2].plot(log_z2)
    axs[2].set_title("Spikes")
    axs[3].plot(log_evec_v)
    axs[3].set_title("Voltage eligibility")
    # axs[1].axhline(y=3, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[1].axhline(y=-3.7, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[1].set_ylim((-4, 3.3))
    axs[4].plot(log_evec_u)
    axs[4].set_title("Refractory eligibility")
    # axs[4].axhline(y=0.005, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[4].axhline(y=-0.002, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[2].set_ylim((-0.0023, 0.0053))
    axs[5].plot(log_h2)
    axs[5].set_title("Pseudo-derivative")
    # axs[5].axhline(y=.3, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[5].axhline(y=0, c='black', alpha=0.3, linestyle='--', linewidth=1)

    axs[6].plot(log_etrace)
    axs[6].set_title("Eligibility trace")
    # axs[6].axhline(y=.88, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[6].axhline(y=-.65, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[3].set_ylim((-.7, .93))
    axs[7].plot(log_grad)
    axs[7].set_title("Gradient")
    # axs[4].axhline(y=245, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[4].axhline(y=183, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[4].set_ylim((180, 248))

    plt.show()


traub_izh()

# def traub_lif():
#     fig, axs = plt.subplots(6, 1)

#     log_v1 = []
#     log_v2 = []
#     log_evec = []
#     log_h2 = []
#     log_etrace = []
#     log_grad = []

#     s1 = 0
#     s2 = 0
#     z1 = 0
#     z2 = 0
#     ev = 0
#     et = 0
#     grad = 0

#     T = 1000
#     tzi = 0
#     tzo = 0
#     vthr = 30
#     alpha = 0.9
#     gamma = 0.3
#     refr = 2

#     A = 4
#     B = 10
#     C = 20
#     for t in range(0, T):
#         if t < T * 0.45:
#             I_i = U(1, B)
#             I_o = U(1, A)
#             if tzi > tzo:
#                 I_o = U(1, C)
#         else:
#             I_i = U(1, A)
#             I_o = U(1, B)
#             if tzo > tzi:
#                 I_i = U(1, C)

#         s1 = alpha * s1 + I_i - z1*alpha*s1 - int(bool(t-tzi == refr))*alpha*s1
#         s2 = alpha * s2 + I_i - z2*alpha*s2 - int(bool(t-tzo == refr))*alpha*s2

#         ev = alpha*(1-z2-int(bool(t-tzo == refr))) * ev + z1

#         h2 = -gamma if t-tzo < refr else gamma * max(0, 1-abs((s2-vthr)/vthr))

#         et = h2*ev




#         log_v1.append(s1)
#         log_v2.append(s2)
#         log_evec.append(ev)
#         log_etrace.append(et)
#         log_grad.append(grad)
#         log_h2.append(h2)

#     axs[0].plot(log_v1)
#     axs[0].plot(log_v2)
#     axs[0].axhline(y=117, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     axs[0].axhline(y=-80, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     axs[0].set_title("Voltage")
#     # axs[0].set_ylim((-100, 130))
#     axs[1].plot(log_evec)
#     axs[1].set_title("Voltage eligibility")
#     axs[1].axhline(y=3, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     axs[1].axhline(y=-3.7, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     # axs[1].set_ylim((-4, 3.3))
#     # axs[2].plot(log_evec_u)
#     # axs[2].set_title("Refractory eligibility")
#     # axs[2].axhline(y=0.005, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     # axs[2].axhline(y=-0.002, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     # axs[2].set_ylim((-0.0023, 0.0053))
#     axs[3].plot(log_h2)
#     axs[2].set_title("Pseudo-derivative")
#     axs[3].axhline(y=.3, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     axs[3].axhline(y=0, c='black', alpha=0.3, linestyle='--', linewidth=1)

#     axs[4].plot(log_etrace)
#     axs[4].set_title("Eligibility trace")
#     axs[4].axhline(y=.88, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     axs[4].axhline(y=-.65, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     # axs[3].set_ylim((-.7, .93))
#     axs[5].plot(log_grad)
#     axs[5].set_title("Gradient")
#     # axs[4].axhline(y=245, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     # axs[4].axhline(y=183, c='black', alpha=0.3, linestyle='--', linewidth=1)
#     # axs[4].set_ylim((180, 248))
#     plt.show()

# traub_lif()
