cfg = {
    "neuron": "LIF",

    "alpha": 0.9,
    "beta": 0.07,
    "gamma": 0.3,
    "rho": 0.995,
    "eqb": -65,
    "thr": 30,
    "dt_refr": 4,

    "refr1": 2,
    "refr2": 0.004,
    "refr3": 0.02,
    "volt1": 0.04,
    "volt2": 5,
    "volt3": 140,

    "dt": .1,

    "N_I": 2,
    "N_R": 3,
    "N_O": 2,
    "N_Rec": 3,

    "Epochs": 100,
    "EMA": 0.6321,

    "plot_io": True,
    "plot_pair": True,
    "plot_heatmaps": False,
    "plot_graph": True,

    "task": "narma10"
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
}
