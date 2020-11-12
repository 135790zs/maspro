from numpy import exp
cfg = {
    "neuron": "ALIF",

    "fraction_ALIF": 1,  # If neuron == LIF
    "theta_adaptation": 200,  # Depends on length of task: working memory

    "theta_membrane": 3,
    "beta": 0.07,
    "gamma": 0.3,     # Pseudoderivative ET contribution
    "eta": 1e-9,      # Learning rate (1e-2 for TIMIT)
    "weight_decay": 0.9,
    "L2_reg": 1e-5,
    "FR_reg": 1e-5,
    "update_dead_weights": False,

    "eqb": -65,       # Voltage equilibrium
    "thr": 30,        # Spike threshold
    "dt_refr": 10,    # Refractory time

    "dt": .1,

    "N_I": 39,  # timit: 39
    "N_R": 63,
    "N_O": 61,  # timit: 61
    "N_Rec": 1,

    "TIMIT_ntrain": 12,  # def 3696
    "TIMIT_nval": 14,  # def 400

    "wavs_fname": "../data_wavs",
    "phns_fname": "../data_phns",
    "weights_fname": "../checkpoint",

    "Epochs": 20,  # def = 80
    "Repeats": 5,  # ms per epoch, def = 5
    "batch_size": 6,  # def = 32
    "maxlen": 200,  # Don't forget to re-run process_timit.py!
    "n_examples": {'train': 30, 'val': 10, 'test': 10},
    "plot_interval": 1,  # 0 to disable plots
    "plot_main": True,
    "plot_state": True,
    "plot_graph": False,
}

cfg['rho'] = exp(-cfg["dt"] / cfg["theta_adaptation"])
cfg['alpha'] = exp(-cfg["dt"] / cfg["theta_membrane"])
cfg['kappa'] = exp(-cfg["dt"] / cfg["theta_membrane"])

assert cfg["N_R"] >= cfg["N_I"]
assert cfg["N_R"] >= cfg["N_O"]

lookup = {
    "X":      {"scalar": False, "binary":False, "label": "x"},
    "V":      {"scalar": False, "binary":False, "label": "v"},
    "Z":      {"scalar": False, "binary":True,  "label": "z"},
    "Z_in":   {"scalar": False, "binary":False, "label": "z_{{in}}"},
    "Zbar":   {"scalar": False, "binary":False, "label": "\\bar{{z}}"},
    "ZbarK":  {"scalar": False, "binary":False, "label": "\\bar{{z}}_\\kappa"},
    "I":      {"scalar": False, "binary":False, "label": "I"},
    "U":      {"scalar": False, "binary":False, "label": "u"},
    "EVV":    {"scalar": False, "binary":False, "label": "\\epsilon_v"},
    "EVU":    {"scalar": False, "binary":False, "label": "\\epsilon_u"},
    "H":      {"scalar": False, "binary":False, "label": "\\psi"},
    "ET":     {"scalar": False, "binary":False, "label": "e"},
    "ETbar":  {"scalar": False, "binary":False, "label": "\\bar{{e}}"},
    "B":      {"scalar": False, "binary":False, "label": "B"},
    "Y":      {"scalar": False, "binary":False, "label": "Y"},
    "P":      {"scalar": False, "binary":False, "label": "\\pi"},
    "Pmax":   {"scalar": False, "binary":True,  "label": "\\pi_{{max}}"},
    "CE":     {"scalar": True,  "binary":False, "label": "CE"},
    "loss":   {"scalar": False, "binary":False, "label": "loss"},
    "T":      {"scalar": False, "binary":True,  "label": "T"},
    "W":      {"scalar": False, "binary":False, "label": "W"},
    "W_rec":  {"scalar": False, "binary":False, "label": "W_{{rec}}"},
    "W_out":  {"scalar": False, "binary":False, "label": "W_{{out}}"},
    "b_out":  {"scalar": False, "binary":False, "label": "b_{{out}}"},
    "DW":     {"scalar": False, "binary":False, "label": "\\Delta W"},
    "Db_out": {"scalar": False, "binary":False, "label": "\\Delta b_{{out}}"},
    "DW_out": {"scalar": False, "binary":False, "label": "\\Delta W_{{out}}"},
}
