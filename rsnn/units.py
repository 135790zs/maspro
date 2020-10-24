import sys
import numpy as np
from matplotlib import rcParams as rc
from config import cfg
import utils as ut
import vis
from rsnn import run_rsnn

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'
np.set_printoptions(threshold=sys.maxsize)


# def simulate_pair(neuron):
#     cfg0 = cfg
#     cfg0["neuron"] = neuron
#     cfg0["N_R"] = 1
#     cfg0["N_Rec"] = 2
#     run_rsnn(cfg=cfg0, layers=(0, 1), neurons=(0, 0))


# def simulate_neurons(model, T=1000, num=2, uses_weights=True):

#     M = {}  # Following order is order of plotting

#     if model == "Izhikevich":
#         M['V'] = np.ones(shape=(T, num,)) * cfg["eqb"]
#     elif model in ["LIF", "ALIF"]:
#         M['V'] = np.zeros(shape=(T, num,))

#     if model in ["LIF", "ALIF"]:
#         X = ut.get_artificial_input(T=T,
#                                     num=num,
#                                     dur=18,
#                                     diff=5,
#                                     interval=100,
#                                     val=3.55,
#                                     switch_interval=500)

#     elif model == "Izhikevich":
#         X = ut.get_artificial_input(T=T,
#                                     num=num,
#                                     dur=30,
#                                     diff=8,
#                                     interval=100,
#                                     val=32,
#                                     switch_interval=500)

#     M['Z'] = np.zeros(shape=(T, num,))

#     if model == "ALIF":
#         M['U'] = np.ones(shape=(T, num,)) * cfg["thr"]
#     elif model in ["LIF", "Izhikevich"]:
#         M['U'] = np.zeros(shape=(T, num,))

#     M['H'] = np.zeros(shape=(T, num,))
#     M['TZ'] = np.ones(shape=(T, num,)) * -cfg["dt_refr"]

#     M['EVV'] = np.zeros(shape=(T, num, num,))
#     M['EVU'] = np.zeros(shape=(T, num, num,))
#     M['ET'] = np.zeros(shape=(T, num, num,))

#     rng = np.random.default_rng()
#     M['W'] = rng.random(size=(T, num, num,))
#     M['L'] = np.ones(shape=(T, num))
#     np.fill_diagonal(M['W'][0, :, :], 0.)
#     Mt = {}

#     for t in range(1, T):

#         for key, item in M.items():
#             Mt[key] = item[t-1]

#         Mt = ut.eprop(
#             model=model,
#             M=Mt,
#             X=X[t-1],
#             t=t,
#             uses_weights=uses_weights)

#         for key, item in Mt.items():
#             M[key][t] = item

#     vis.plot_pair(M=M, ep=T, X=X, is_unit=True)


# simulate_neurons(model=cfg["neuron"], uses_weights=True)
for neuron in ["LIF", "ALIF", "Izhikevich"]:
    cfg0 = cfg
    cfg0["neuron"] = neuron
    cfg0["N_R"] = 1
    cfg0["N_Rec"] = 2
    cfg0["Epochs"] = 500
    cfg0["plot_interval"] = cfg0["Epochs"]
    run_rsnn(cfg=cfg0, layers=(0, 1), neurons=(0, 0), fname=neuron)
