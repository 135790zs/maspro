cfg = {
    "neuron": "LIF",

    "alpha": 0.9,  # Leak factor
    "beta": 1,
    "gamma": 0.3,  # Pseudoderivative ET contribution
    "eta": 0.01,  # Learning rate
    "kappa": 0.5,  # Output smoothing
    "rho": 0.99,  # Threshold leakage
    "eqb": -65,
    "thr": 30,
    "dt_refr": 10,

    "refr1": 2,
    "refr2": 0.004,
    "refr3": 0.02,
    "volt1": 0.04,
    "volt2": 5,
    "volt3": 140,

    "dt": .1,

    "N_R": 3,
    "N_Rec": 3,

    "Epochs": 1000,
    "EMA": 0.05,

    "plot_interval": 1,  # 0 to disable plots
    "plot_io": True,
    "plot_pair": True,
    "plot_heatmaps": False,
    "plot_graph": True,

    "task": "pulse"
}

lookup = {
    "X":   {"dim": 2, "label": "x^t"},
    "V":   {"dim": 2, "label": "v^t"},
    "Z":   {"dim": 2, "label": "z^t"},
    "U":   {"dim": 2, "label": "u^t"},
    "EVV": {"dim": 3, "label": "\\epsilon^t"},
    "EVU": {"dim": 3, "label": "\\epsilon^t"},
    "H":   {"dim": 2, "label": "h^t"},
    "ET":  {"dim": 3, "label": "e^t"},
    "W":   {"dim": 3, "label": "W^t"},
    "DW":  {"dim": 3, "label": "\\Delta W"}
}
