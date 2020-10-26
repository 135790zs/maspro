cfg = {
    "neuron": "ALIF",

    "alpha": 0.95,  # Leak factor
    "beta": 1,
    "gamma": 0.3,   # Pseudoderivative ET contribution
    "eta": 0.01,    # Learning rate
    "kappa": 0.5,   # Output smoothing
    "rho": 0.99,    # Threshold leakage
    "eqb": -65,     # Voltage equilibrium
    "thr": 30,      # Spike threshold
    "dt_refr": 10,  # Refractory time

    "refr1": 2,
    "refr2": 0.004,
    "refr3": 0.02,  # Refractory decay
    "volt1": 0.04,
    "volt2": 5,
    "volt3": 140,

    "dt": .1,

    "N_R": 16,
    "N_Rec": 1,

    "Epochs": 2000,
    "EMA": 0.05,

    "plot_interval": 200,  # 0 to disable plots
    "plot_io": False,
    "plot_state": True,
    "plot_heatmaps": False,
    "plot_graph": False,

    "task": "pulse"
}

lookup = {
    "X":   {"dim": 2, "label": "x"},
    "V":   {"dim": 2, "label": "v"},
    "Z":   {"dim": 2, "label": "z"},
    "I":   {"dim": 2, "label": "I"},
    "U":   {"dim": 2, "label": "u"},
    "EVV": {"dim": 3, "label": "\\epsilon_v"},
    "EVU": {"dim": 3, "label": "\\epsilon_u"},
    "H":   {"dim": 2, "label": "h"},
    "ET":  {"dim": 3, "label": "e"},
    "W":   {"dim": 3, "label": "W"},
    "DW":  {"dim": 3, "label": "\\Delta W"}
}
