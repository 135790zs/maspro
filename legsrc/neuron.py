import numpy as np


def Vm_diff_t(Vm_t,
              g_ej_t,
              g_ij_t,
              Rm=100,
              Cm=0.3,
              V_resting=-70,
              E_e=0,
              E_i=-75,
              I_noise=0):
    term1 = - (Vm_t - V_resting) / Rm
    term2 = - np.sum(g_ej_t * (Vm_t - E_e))
    term3 = - np.sum(g_ij_t * (Vm_t - E_i))
    term4 = - I_noise

    return (term1 + term2 + term3 + term4) / Cm


n_neurons = 5
total_time = 4

neurons = np.zeros(shape=(n_neurons,))
synapse = np.random.random(size=(n_neurons, n_neurons))

log = []

V_thresh = -59
V_reset = -70

for time in range(total_time):

    log.append(neurons[1])
    for idx, neuron in enumerate(neurons):
        d = Vm_diff_t(Vm_t=neuron,
                      g_ej_t=synapse[idx, :],
                      g_ij_t=synapse[idx, :])
        print(d)
        neurons[idx] += d
        if neurons[idx] >= V_thresh:
            neurons[idx] = V_reset

print(log)


