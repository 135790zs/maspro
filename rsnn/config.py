from numpy import exp
cfg = {
    "verbose": True,
    "eprop_type": "random",  # in {global, random, symmetric, adaptive}
    "optimizer": 'Adam',  # in {Adam, SGD}
    "traub_trick": False,
    "fraction_ALIF": 0.25,  # def 0.25
    "n_directions": 1,
    "delay": 0,
    "seed": 42,  # 'None' for random seed
    "load_checkpoints": None,

    "alpha": 0.779,  # Bellec1: 20 = 0.951. 4: 0.779
    "rho": 0.975,  # Bellec1: 200 = 0.995. 40: 0.975
    "kappa": 0.779,  # Bellec1: 3 = 0.717. .75:~0.25
    "beta": 1.8,    # Bellecs: 1.8. Adaptive strength. 0 = LIF
    "gamma": 0.3,     # Bellecs: 0.3.
    "eta_b_out": None,  # None=no sep. eta. Otherwise=Constant
    "eta_init": 1e-2,   # Bellecs: 1e-2
    "eta_slope": 0.8,      # Slope defining relation between Verr and eta (1e-2 for TIMIT)
    "eta_init_loss": 0,  # 0 to disable annealing. This is the cap below which annealing to 0 takes place.
    "ramping": 0,  # Ramps eta linearly from 0 to init between epoch 0 and this one. 0 to disable.
    "thr": 1,        # Bellec1: unknown. Bellec2: 1.6?
    "dt_refr": 2,    # Bellec1: 2
    "weight_decay": 1e-2,  # Bellec1: 0. Bellec2: 1e-2. For W_out and B, only if adaptive.
    "L2_reg": 0,  # Bellec1: 0. Bellec2: 1e-5
    "FR_target": 1e-2,  # Bellecs: 1e-2
    "FR_reg": 1,  # Bellec1: 1. Bellec2: 50.
    "dropout": 0.8,  # of recurrent (excl inputs)
    "softmax_factor": 1,  # Bellecs: 1
    "weight_scaling": 1,  # Bellecs: 1
    "uniform_weights": True,
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

    "N_R": 200,
    "N_Rec": 1,

    "task": "TIMIT",
    "wavs_fname": "../data/data_wavs",
    "phns_fname": "../data/data_phns",

    "Epochs": 80,  # def = 80
    "Track_weights": True,
    "Track_synapse": False,  # Only for nonweight synapse vars (e.g. ET)
    "Repeats": 1,  # ms per epoch, def = 5
    "batch_size_train": 10,  # def = 32
    "batch_size_val": 10,  # def = 32
    "batch_size_test": 32,  # def = 32
    "val_every_E": 5,
    "maxlen": 778,  #def 778, Don't forget to re-run process_timit.py!
    "TIMIT_derivative": 0,
    "n_examples": {'train': 20, 'val': 20, 'test': 2},
    # "n_examples": {'train': 3696, 'val': 400, 'test': 192},
    "plot_state_interval": 5,  #  State plot; 0 to disable plots
    "state_save_interval": 1,
    "plot_run_interval": 5,
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
    "Z_prev":  {"scalar": False, "binary":True,  "label": "z_{{prev}}"},
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
