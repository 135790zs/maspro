cfg = {
    "neuron": "ALIF",

    "fraction_ALIF": 1.0,  # If neuron == LIF

    "alpha": 0.99,    # Leak factor
    "beta": 0.07,
    "gamma": 0.3,     # Pseudoderivative ET contribution
    "eta": 1e-10,    # Learning rate
    "kappa": 0.995,    # Output smoothing
    "rho": 0.999,      # Threshold leakage
    "eqb": -65,       # Voltage equilibrium
    "thr": 30,        # Spike threshold
    "dt_refr": 100,    # Refractory time

    # "refr1": 2,
    # "refr2": 0.004,
    # "refr3": 0.02,  # Refractory decay
    # "volt1": 0.04,
    # "volt2": 5,
    # "volt3": 140,

    "dt": .1,

    "N_I": 1,
    "N_R": 2,
    "N_Rec": 2,

    "Epochs": 30,

    "plot_interval": 30,  # 0 to disable plots
    "plot_io": False,
    "plot_state": True,
    "plot_heatmaps": False,
    "plot_graph": True,

    "task": "pulse"
}

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
