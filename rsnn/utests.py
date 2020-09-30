from config import config
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

# Traub alg 2

rng = np.random.default_rng()


def U(a, b):
    return a + rng.random() * (b - a)


def traub_izh():
    fig, axs = plt.subplots(7, 1)

    log_v1 = []
    log_v2 = []
    log_evec_v = []
    log_evec_u = []
    log_h2 = []
    log_etrace = []
    log_grad = []
    log_u1 = []
    log_u2 = []

    s1 = np.asarray([-65., -10.])
    s2 = np.asarray([-65., -10.])
    z1 = 0
    z2 = 0
    ev = np.zeros(shape=(2,))
    et = 0
    grad = 0

    T = 10000
    tzi = 0
    tzo = 0
    A = 8
    C = 10
    for t in range(0, T):
        if t < T * 0.45:
            I_i = A
            I_o = 0
            if tzi > tzo:
                I_o = C
        else:
            I_i = 0
            I_o = A
            if tzo > tzi:
                I_i = C

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

        evvnext = (1 - z2) * (1 + (config["EVV1"]*s2[0] + config["EVV2"]) * config["dt"]) * ev[0] - config["dt"] * ev[1] + config["dt"] * z1

        ev[1] = config["EVU1"] * config["dt"] * (1 - z2) * ev[0] + (1 - config["EVU2"] * config["dt"]) * ev[1]
        ev[0] = evvnext

        s1v = ut.vnext(v=s1[0], u=s1[1], z=z1, I=I_i)
        s1u = ut.unext(v=s1[0], u=s1[1], z=z1)
        s2v = ut.vnext(v=s2[0], u=s2[1], z=z2, I=I_o)
        s2u = ut.unext(v=s2[0], u=s2[1], z=z2)
        s1[0] = s1v
        s1[1] = s1u
        s2[0] = s2v
        s2[1] = s2u
        h2 = ut.h(v=s2[0])
        et = h2 * ev[0]

        grad = grad + 10 * et

        # z1 = z1n
        # z2 = z2n

        log_v1.append(s1[0])
        log_v2.append(s2[0])
        log_evec_v.append(ev[0])
        log_evec_u.append(ev[1])
        log_etrace.append(et)
        log_grad.append(grad)
        log_h2.append(h2)
        log_u1.append(s1[1])
        log_u2.append(s2[1])
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
    axs[1].plot(log_evec_v)
    axs[1].set_title("Voltage eligibility")
    axs[1].axhline(y=3, c='black', alpha=0.3, linestyle='--', linewidth=1)
    axs[1].axhline(y=-3.7, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[1].set_ylim((-4, 3.3))
    axs[2].plot(log_evec_u)
    axs[2].set_title("Refractory eligibility")
    axs[2].axhline(y=0.005, c='black', alpha=0.3, linestyle='--', linewidth=1)
    axs[2].axhline(y=-0.002, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[2].set_ylim((-0.0023, 0.0053))
    axs[3].plot(log_h2)
    axs[2].set_title("Pseudo-derivative")
    axs[3].axhline(y=.3, c='black', alpha=0.3, linestyle='--', linewidth=1)
    axs[3].axhline(y=0, c='black', alpha=0.3, linestyle='--', linewidth=1)

    axs[4].plot(log_etrace)
    axs[4].set_title("Eligibility trace")
    axs[4].axhline(y=.88, c='black', alpha=0.3, linestyle='--', linewidth=1)
    axs[4].axhline(y=-.65, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[3].set_ylim((-.7, .93))
    axs[5].plot(log_grad)
    axs[5].set_title("Gradient")
    # axs[4].axhline(y=245, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[4].axhline(y=183, c='black', alpha=0.3, linestyle='--', linewidth=1)
    # axs[4].set_ylim((180, 248))

    axs[6].plot(log_u1)
    axs[6].plot(log_u2)
    axs[0].set_title("Refractory")
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
