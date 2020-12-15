from numpy import exp
cfg = {
    "verbose": True,
    "eprop_type": "adaptive",  # in {random, symmetric, adaptive}
    "optimizer": 'SGD',  # in {Adam, SGD}
    "traub_trick": False,
    "fraction_ALIF": 0.25,  # def 0.25
    "n_directions": 1,
    "delay": 0,

    "rho": .995,  # Depends on length of task: working memory. TIMIT= .995
    "alpha": 0.95,  # TIMIT: .95
    "kappa": 0.717,  # TIMIT: .717
    "beta": 0.184,    # TIMIT: 0.184
    "gamma": 0.45,     # Pseudoderivative ET contribution
    "eta_init": 1e-2,   # Initial learning rate (1e-2 for TIMIT)
    "eta_slope": 2,      # Slope defining relation between Verr and eta (1e-2 for TIMIT)
    "eta_init_loss": 0,  # 0 to disable annealing. This is the cap below which annealing to 0 takes place linearly.
    "thr": 1.6,        # Spike threshold, def = 1.6
    "dt_refr": 2,    # Refractory time, def = 2
    "weight_decay": 1e-2,  # For W_out and B, only if adaptive. def = 1e-2
    "L2_reg": 1e-5,  # 1e-5 for TIMIT
    "FR_target": 0.01,  # Desired frequency (mean spike per ms)
    "FR_reg": 50,  # 50 for TIMIT
    "dropout": 0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-5,

    "update_bias": False,
    "update_W_out": False,
    "update_W": True,
    "update_dead_weights": False,
    "update_input_weights": True,  # Subset of - and overridden by `update_W'.

    "N_R": 64,
    "N_Rec": 1,

    "wavs_fname": "../data_wavs",
    "phns_fname": "../data_phns",

    "Epochs": 40,  # def = 80
    "Track_weights": True,
    "Track_state": False,  # Only for synapse vars
    "Repeats": 5,  # ms per epoch, def = 5
    "TIMIT_derivative": 0,
    "batch_size_train": 6,  # def = 32
    "batch_size_val": 6,  # def = 32
    "val_every_E": 2,
    "maxlen": 778,  # Don't forget to re-run process_timit.py!
    "n_examples": {'train': 3696, 'val': 400, 'test': 192},
    # "n_examples": {'train': 30, 'val': 30, 'test': 30},
    "plot_interval": 1,  #  State plot; 0 to disable plots
    "state_save_interval": 5,
    "plot_main": True,
    "plot_state": True,
    "plot_graph": False,

    "frame_size": 0.025,
    "frame_stride": 0.01,
    "pre_emphasis": 0.97,
    "nfilt": 26,
    "NFFT": 512,
    "num_ceps": 12,
    "cep_lifter": 22,
}

lookup = {
    "X":       {"scalar": False, "binary":False, "label": "x"},
    "X1":      {"scalar": False, "binary":False, "label": "x_{{fwd}}"},
    "X2":      {"scalar": False, "binary":False, "label": "x_{{rev}}"},
    "V":       {"scalar": False, "binary":False, "label": "v"},
    "Z":       {"scalar": False, "binary":True,  "label": "z"},
    "Z_in":    {"scalar": False, "binary":True,  "label": "z_{{in}}"},
    "Z_inbar": {"scalar": False, "binary":False, "label": "\\bar{{z}}_{{in}}"},
    "Zbar":    {"scalar": False, "binary":False, "label": "\\bar{{z}}_\\kappa"},
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
