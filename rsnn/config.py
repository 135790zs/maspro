from numpy import exp
cfg = {
    "neuron": "ALIF",

    "fraction_ALIF": 1,  # If neuron == LIF
    "theta_adaptation": 200,  # Depends on length of task: working memory

    "theta_membrane": 3,
    "beta": 0.07,
    "gamma": 0.3,     # Pseudoderivative ET contribution
    "eta": 1e-2,      # Learning rate
    "weight_decay": 0.9,
    "update_dead_weights": True,

    "eqb": -65,       # Voltage equilibrium
    "thr": 30,        # Spike threshold
    "dt_refr": 100,    # Refractory time

    # "refr1": 2,
    # "refr2": 0.004,
    # "refr3": 0.02,  # Refractory decay
    # "volt1": 0.04,
    # "volt2": 5,
    # "volt3": 140,

    "dt": 1,

    "N_I": 3,
    "N_R": 4,
    "N_O": 4,
    "N_Rec": 1,

    "Epochs": 30,
    "plot_interval": 30,  # 0 to disable plots
    "plot_io": False,
    "plot_state": True,
    "plot_heatmaps": False,
    "plot_graph": True,

    "task": "pulseclass",
    "tasktype": "classification",  # classification, regression
}

cfg['rho'] = exp(-cfg["dt"] / cfg["theta_adaptation"])
cfg['alpha'] = 0.95#exp(-cfg["dt"] / cfg["theta_membrane"])
cfg['kappa'] = exp(-cfg["dt"] / cfg["theta_membrane"])

assert cfg["N_R"] >= cfg["N_I"]
assert cfg["N_R"] >= cfg["N_O"]

lookup = {
    "X":     {"dim": 2, "label": "x"},
    "XZ":    {"dim": 2, "label": "x_{{z}}"},
    "V":     {"dim": 2, "label": "v"},
    "Z":     {"dim": 2, "label": "z"},
    "Z_in":  {"dim": 2, "label": "z_{{in}}"},
    "I":     {"dim": 2, "label": "I"},
    "U":     {"dim": 2, "label": "u"},
    "EVV":   {"dim": 3, "label": "\\epsilon_v"},
    "EVU":   {"dim": 3, "label": "\\epsilon_u"},
    "H":     {"dim": 2, "label": "h"},
    "ET":    {"dim": 3, "label": "e"},
    "B":     {"dim": 3, "label": "B"},
    "Y":     {"dim": 3, "label": "Y"},
    "error": {"dim": 3, "label": "E"},
    "loss":  {"dim": 3, "label": "loss"},
    "T":     {"dim": 3, "label": "T"},
    "W":     {"dim": 3, "label": "W"},
    "W_out": {"dim": 1, "label": "W_{{out}}"},
    "b_out": {"dim": 0, "label": "b_{{out}}"},
    "DW":    {"dim": 3, "label": "\\Delta W"}
}


"""
TO DO ON OCT 28:
* Regularization
* Easiest Bellec task?
* ... Bellec steps

"""
