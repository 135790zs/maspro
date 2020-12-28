from numpy import exp
cfg = {
    "verbose": True,
    "eprop_type": "random",  # in {global, random, symmetric, adaptive}
    "optimizer": 'Adam',  # in {Adam, SGD}
    "traub_trick": False,
    "fraction_ALIF": 1,  # def 0.25
    "n_directions": 1,
    "delay": 0,

    "alpha": exp(-1/20),  # Bellec1: 20 = 0.951
    "rho": exp(-1/200),  # Bellec1: 200 = 0.995
    "kappa": exp(-1/3),  # Bellec1: 3 = 0.717
    "beta": 1,    # Bellecs: 1.8. Adaptive strength. 0 = LIF
    "gamma": 0.3,     # Bellecs: 0.3.
    "eta_b_out": None,  # Constant
    "eta_init": 1e-2,   # Bellecs: 1e-2
    "eta_slope": 1,      # Slope defining relation between Verr and eta (1e-2 for TIMIT)
    "eta_init_loss": 0,  # 0 to disable annealing. This is the cap below which annealing to 0 takes place.
    "ramping": 3,  # Ramps eta linearly from 0 to init between epoch 0 and this one. 0 to disable.
    "thr": 5,        # Bellec1: unknown. Bellec2: 1.6?
    "dt_refr": 2,    # Bellec1: 2
    "weight_decay": 0,  # Bellec1: 0. Bellec2: 1e-2. For W_out and B, only if adaptive.
    "L2_reg": 0,  # Bellec1: 0. Bellec2: 1e-5
    "FR_target": 1e-2,  # Bellecs: 1e-2
    "FR_reg": 50,  # Bellec1: 1. Bellec2: 50.
    "dropout": 0.8,  # of recurrent (excl inputs)
    "softmax_factor": 1,  # Bellecs: 1
    "weight_scaling": 1,  # Bellecs: 1
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-5,

    "one_to_one_input": False,
    "update_input_weights": True,  # Subset of - and overridden by `update_W'.
    "update_W": True,
    "update_bias": True,
    "update_W_out": True,
    "one_to_one_output": False,
    "update_dead_weights": False,

    "N_R": 92,
    "N_Rec": 1,

    "task": "TIMIT_small",
    "wavs_fname": "../data/data_wavs",
    "phns_fname": "../data/data_phns",

    "Epochs": 150,  # def = 80
    "Track_weights": True,
    "Track_synapse": False,  # Only for synapse vars
    "Repeats": 5,  # ms per epoch, def = 5
    "batch_size_train": 5,  # def = 32
    "batch_size_val": 5,  # def = 32
    "batch_size_test": 32,  # def = 32
    "val_every_E": 5,
    "maxlen": 778,  #def 778, Don't forget to re-run process_timit.py!
    "TIMIT_derivative": 0,
    "n_examples": {'train': 10, 'val': 10, 'test': 2},
    # "n_examples": {'train': 3696, 'val': 400, 'test': 192},
    "plot_interval": 1,  #  State plot; 0 to disable plots
    "state_save_interval": 20,
    "plot_main": True,
    "plot_state": True,
    "plot_graph": False,

    "frame_size": 0.025,
    "frame_stride": 0.01,
    "pre_emphasis": 0.97,
    "nfilt": 26,
    "NFFT": 512,
    "num_ceps": 13,
    "cep_lifter": 22,
}

lookup = {
    "X":       {"scalar": False, "binary":False, "label": "x"},
    "X1":      {"scalar": False, "binary":False, "label": "x_{{fwd}}"},
    "X2":      {"scalar": False, "binary":False, "label": "x_{{rev}}"},
    "V":       {"scalar": False, "binary":False, "label": "v"},
    "Z":       {"scalar": False, "binary":True,  "label": "z"},
    "TZ":      {"scalar": False, "binary":False, "label": "TZ"},
    "Z_in":    {"scalar": False, "binary":True,  "label": "z_{{in}}"},
    "TZ_in":   {"scalar": False, "binary":False, "label": "TZ_{{in}}"},
    "Z_inbar": {"scalar": False, "binary":False, "label": "\\bar{{z}}_{{in}}"},
    "Zbar":    {"scalar": False, "binary":False, "label": "\\bar{{z}}_\\kappa"},
    "I":       {"scalar": False, "binary":False, "label": "I"},
    "U":       {"scalar": False, "binary":False, "label": "a"},
    "EVV":     {"scalar": False, "binary":False, "label": "\\epsilon_v"},
    "EVU":     {"scalar": False, "binary":False, "label": "\\epsilon_a"},
    "H":       {"scalar": False, "binary":False, "label": "\\psi"},
    "ET":      {"scalar": False, "binary":False, "label": "e"},
    "L_std":   {"scalar": False, "binary":False, "label": "L_{{std}}"},
    "L_reg":   {"scalar": False, "binary":False, "label": "L_{{reg}}"},
    "ETbar":   {"scalar": False, "binary":False, "label": "\\bar{{e}}"},
    "B":       {"scalar": False, "binary":False, "label": "B"},
    "Y":       {"scalar": False, "binary":False, "label": "Y"},
    "P":       {"scalar": False, "binary":False, "label": "\\pi"},
    "D":       {"scalar": False, "binary":True,  "label": "D"},
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
    "gW":      {"scalar": False, "binary":False, "label": "\\nabla W"},
    "Db_out":  {"scalar": False, "binary":False, "label": "\\Delta b_{{out}}"},
    "DW_out":  {"scalar": False, "binary":False, "label": "\\Delta W_{{out}}"},
    "DW_reg":  {"scalar": False, "binary":False, "label": "\\Delta W_{{reg}}"},
}
