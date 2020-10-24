import sys
import numpy as np
from matplotlib import rcParams as rc
from config import cfg
from rsnn import run_rsnn

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'
np.set_printoptions(threshold=sys.maxsize)

for neuron in ["LIF", "ALIF", "Izhikevich"]:
    cfg0 = cfg
    cfg0["neuron"] = neuron
    cfg0["N_R"] = 1
    cfg0["N_Rec"] = 2
    cfg0["Epochs"] = 500
    cfg0["plot_interval"] = cfg0["Epochs"]
    run_rsnn(cfg=cfg0, layers=(0, 1), neurons=(0, 0), fname=neuron)
