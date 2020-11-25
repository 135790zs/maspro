from numpy import exp
cfg = {
    "eprop_type": "adaptive",  # in {random, symmetric, adaptive}
    "optimizer": 'SGD',
    "fraction_ALIF": 0.25,  # def 0.25
    "theta_adaptation": 80,  # Depends on length of task: working memory
    "n_directions": 2,
    "delay": 0,

    "theta_membrane": 4,  # TIMIT: 20
    "theta_output": 0.75,  # TIMIT: 3
    "beta": 0.184,    # TIMIT: 0.184
    "gamma": 0.3,     # Pseudoderivative ET contribution
    "eta": 1e-4,      # Learning rate (1e-2 for TIMIT)
    "thr": 1.6,        # Spike threshold, def = 1.6
    "dt_refr": 2,    # Refractory time, def = 2
    "dt": 1,
    "weight_decay": 1e-2,  # For W_out and B, only if adaptive. def = 1e-2
    "L2_reg": 1e-5,  # 1e-5 for TIMIT
    "FR_reg": 50,  # 50 for TIMIT
    "FR_target": 6/1000,  # Desired frequency (mean per ms)
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-5,


    "update_bias": True,
    "update_W_out": True,
    "update_dead_weights": False,
    "update_input_weights": False,

    "N_R": 128,
    "N_Rec": 1,

    "wavs_fname": "../data_wavs",
    "phns_fname": "../data_phns",
    "weights_fname": "../checkpoint",

    "Epochs": 5000,  # def = 80
    "Repeats": 2,  # ms per epoch, def = 5
    "batch_size_train": 32,  # def = 32
    "batch_size_val": 32,  # def = 32
    "val_every_E": 10,
    "maxlen": 300,  # Don't forget to re-run process_timit.py!
    "n_examples": {'train': 800, 'val': 400, 'test': 400},
    "plot_interval": 10,  #  State plot; 0 to disable plots
    "plot_main": True,
    "plot_state": True,
    "plot_graph": False,
}

cfg['rho'] = exp(-cfg["dt"] / cfg["theta_adaptation"])
cfg['alpha'] = exp(-cfg["dt"] / cfg["theta_membrane"])
cfg['kappa'] = exp(-cfg["dt"] / cfg["theta_output"])

lookup = {
    "X":       {"scalar": False, "binary":False, "label": "x"},
    "X1":      {"scalar": False, "binary":False, "label": "x_{{fwd}}"},
    "X2":      {"scalar": False, "binary":False, "label": "x_{{rev}}"},
    "V":       {"scalar": False, "binary":False, "label": "v"},
    "Z":       {"scalar": False, "binary":True,  "label": "z"},
    "Z_in":    {"scalar": False, "binary":True,  "label": "z_{{in}}"},
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
