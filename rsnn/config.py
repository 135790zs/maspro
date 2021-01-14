from numpy import exp
cfg = {
    "verbose": True,
    "eprop_type": "random",  # in {global, random, symmetric, adaptive}
    "optimizer": 'Adam',  # in {Adam, SGD}
    "traub_trick": False,  # Change to "neurontype" soon {(A)LIF, Izh, STDP-(A)LIF}
    "v_fix": False,
    "fraction_ALIF": 0.25,  # def 0.25
    "n_directions": 1,  # Reduces error from 36.1 to 32.9.
    "delay": 0,
    "seed": None,  # 'None' for random seed
    "max_duration": None,
    "load_checkpoints": None,

    "beta": 0.184,  # Bellec2: "order of 0.07", Bellec3: 0.184. Code: 1.8
    "gamma": 0.3,  # Bellec2: 0.3
    "thr": 1.6,  # Bellec3: 1.6
    "dt_refr": 2,  # Bellec3: 2
    "weight_initialization": "bellec18",

    "eta_init": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-5,

    "weight_decay": 1e-2,  # Bellec3: 1e-2
    "L2_reg": 0,  # Bellec3: 1e-5
    "FR_target": 0.01,  # BellecCode: 0.01 (10hz)
    "FR_reg": 50,  # Bellec3: 50

    "eta_b_out": None,  # None=no sep. eta. Otherwise=Constant
    "eta_slope": 2,      # Slope defining relation between Verr and eta (1e-2 for TIMIT)
    "eta_init_loss": 0,  # 0 to disable annealing. This is the cap below which annealing to 0 takes place.
    "ramping": 0,  # Ramps eta in first epoch
    "unramp": 0,  # Ramps eta linearly from init to 0 between epoch 0 and this one. 0 to disable.
    "softmax_factor": 1,  # Bellecs: 1
    "weight_scaling": 1,  # Bellecs: 1
    "dropout": 0,  # of recurrent (excl inputs), bellecs = 0

    "update_W": True,
    "update_W_out": True,
    "update_b_out": True,
    "one_to_one_input": False,
    "one_to_one_output": False,
    "update_dead_weights": False,
    "update_input_weights": True,  # Subset of - and overridden by `update_W'.
    "recurrent": True,

    "N_R": 400,
    "N_Rec": 1,

    "task": "TIMIT",
    "wavs_fname": "../data/data_wavs",
    "phns_fname": "../data/data_phns",

    "Epochs": 9500,  # def = 80
    "Track_weights": False,
    "Track_synapse": False,  # Only for nonweight synapse vars (e.g. ET)
    "Repeats": 1,  # ms per epoch, def = 5
    "Interpolation": 'nearest',  # nearest, linear
    "batch_size_train": 32,  # def = 32
    "batch_size_val": 32,  # def = 32
    "batch_size_test": 1,  # def = 32
    "maxlen": 778,  #def 778, Don't forget to re-run process_timit.py!
    "TIMIT_derivative": 0,
    "n_examples": {'train': 12, 'val': 12, 'test': 1},
    # "n_examples": {'train': 3696, 'val': 400, 'test': 192},
    "plot_state_interval": 20,  # Per iter  #  State plot; 0 to disable plots
    "state_save_interval": 20,
    "plot_run_interval": 5,  # Per epoch
    "plot_pair_interval": 0,
    "val_every_E": 5,
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

cfg["alpha"] = exp(-1/(4*cfg['Repeats']))  # Bellec1: 20 = 0.951. 4: 0.779
cfg["rho"] = exp(-1/(40*cfg["Repeats"]))  # Bellec1: 200 = 0.995. 40: 0.975
cfg["kappa"] = exp(-1/(0.6*cfg["Repeats"]))  # Bellec1: 3 = 0.717. .75:~0.25
# cfg["FR_target"] /= cfg["Repeats"]
# cfg["FR_reg"] /= cfg["Repeats"]

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
    "I_in":    {"scalar": False, "binary":False, "label": "I_{{in}}"},
    "I_rec":   {"scalar": False, "binary":False, "label": "I_{{rec}}"},
    "a":       {"scalar": False, "binary":False, "label": "a"},
    "A":       {"scalar": False, "binary":False, "label": "A"},
    "EVV":     {"scalar": False, "binary":False, "label": "\\epsilon_v"},
    "EVU":     {"scalar": False, "binary":False, "label": "\\epsilon_a"},
    "H":       {"scalar": False, "binary":False, "label": "\\psi"},
    "ET":      {"scalar": False, "binary":False, "label": "e"},
    "L":       {"scalar": False, "binary":False, "label": "L"},
    "L_std":   {"scalar": False, "binary":False, "label": "L_{{std}}"},
    "L_reg":   {"scalar": False, "binary":False, "label": "L_{{reg}}"},
    "spikerate":{"scalar": False, "binary":False, "label": "Hz"},
    "ETbar":   {"scalar": False, "binary":False, "label": "\\bar{{e}}"},
    "B":       {"scalar": False, "binary":False, "label": "B"},
    "Y":       {"scalar": False, "binary":False, "label": "Y"},
    "P":       {"scalar": False, "binary":False, "label": "\\pi"},
    "D":       {"scalar": False, "binary":True,  "label": "D"},
    "Pmax":    {"scalar": False, "binary":True,  "label": "\\pi_{{max}}"},
    "CE":      {"scalar": True,  "binary":False, "label": "CE"},
    "Correct": {"scalar": False, "binary":True,  "label": "Correct"},
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
