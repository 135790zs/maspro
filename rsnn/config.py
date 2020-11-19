from numpy import exp
cfg = {
    "neuron": "ALIF",
    "eprop_type": "adaptive",  # in {random, symmetric, adaptive}

    "fraction_ALIF": 1,  # If neuron == LIF
    "theta_adaptation": 200,  # Depends on length of task: working memory

    "theta_membrane": 3,
    "beta": 0.07,
    "gamma": 0.3,     # Pseudoderivative ET contribution
    "eta": 1e-2,      # Learning rate (1e-2 for TIMIT)
    "weight_decay": 1,  # For W_out and B, only if adaptive
    "L2_reg": 0,
    "FR_reg": 1e-5,
    "FR_target": 0.1,
    "update_dead_weights": False,
    "update_input_weights": False,

    "eqb": -65,       # Voltage equilibrium
    "thr": 10,        # Spike threshold
    "dt_refr": 10,    # Refractory time

    "dt": .1,

    "N_R": 2,
    "N_Rec": 1,

    "TIMIT_ntrain": 12,  # def 3696
    "TIMIT_nval": 14,  # def 400

    "wavs_fname": "../data_wavs",
    "phns_fname": "../data_phns",
    "weights_fname": "../checkpoint",

    "Epochs": 200,  # def = 80
    "Repeats": 2,  # ms per epoch, def = 5
    "batch_size": 10,  # def = 32
    "maxlen": 100,  # Don't forget to re-run process_timit.py!
    "n_examples": {'train': 200, 'val': 100, 'test': 100},
    "plot_interval": 1,  # 0 to disable plots
    "plot_main": True,
    "plot_state": True,
    "plot_graph": False,
}

cfg['rho'] = exp(-cfg["dt"] / cfg["theta_adaptation"])
cfg['alpha'] = exp(-cfg["dt"] / cfg["theta_membrane"])
cfg['kappa'] = exp(-cfg["dt"] / cfg["theta_membrane"])

lookup = {
    "X":       {"scalar": False, "binary":False, "label": "x"},
    "V":       {"scalar": False, "binary":False, "label": "v"},
    "Z":       {"scalar": False, "binary":True,  "label": "z"},
    "Z_in":    {"scalar": False, "binary":False, "label": "z_{{in}}"},
    "Z_inbar": {"scalar": False, "binary":False, "label": "\\bar{{z}}_{{in}}"},
    "Zbar":    {"scalar": False, "binary":False, "label": "\\bar{{z}}"},
    "ZbarK":   {"scalar": False, "binary":False, "label": "\\bar{{z}}_\\kappa"},
    "I":       {"scalar": False, "binary":False, "label": "I"},
    "U":       {"scalar": False, "binary":False, "label": "u"},
    "EVV":     {"scalar": False, "binary":False, "label": "\\epsilon_v"},
    "EVU":     {"scalar": False, "binary":False, "label": "\\epsilon_u"},
    "H":       {"scalar": False, "binary":False, "label": "\\psi"},
    "ET":      {"scalar": False, "binary":False, "label": "e"},
    "L":       {"scalar": False, "binary":False, "label": "L"},
    "ETbar":   {"scalar": False, "binary":False, "label": "\\bar{{e}}"},
    "B":       {"scalar": False, "binary":False, "label": "B"},
    "Y":       {"scalar": False, "binary":False, "label": "Y"},
    "P":       {"scalar": False, "binary":False, "label": "\\pi"},
    "Pmax":    {"scalar": False, "binary":True,  "label": "\\pi_{{max}}"},
    "CE":      {"scalar": True,  "binary":False, "label": "CE"},
    "loss":    {"scalar": False, "binary":False, "label": "loss"},
    "T":       {"scalar": False, "binary":True,  "label": "T"},
    "w":       {"scalar": True,  "binary":False, "label": "w^{{rec}}"},
    "W":       {"scalar": False, "binary":False, "label": "W"},
    "W_rec":   {"scalar": False, "binary":False, "label": "W_{{rec}}"},
    "W_out":   {"scalar": False, "binary":False, "label": "W_{{out}}"},
    "b_out":   {"scalar": False, "binary":False, "label": "b_{{out}}"},
    "DW":      {"scalar": False, "binary":False, "label": "\\Delta W"},
    "Db_out":  {"scalar": False, "binary":False, "label": "\\Delta b_{{out}}"},
    "DW_out":  {"scalar": False, "binary":False, "label": "\\Delta W_{{out}}"},
}
