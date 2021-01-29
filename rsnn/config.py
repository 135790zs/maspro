from numpy import exp
cfg = {
    "eprop_type": "random",  # in {global, random, symmetric, adaptive}
    "Optimizer": "Adam",
    "neuron": "STDP-ALIF",  # in {ALIF, STDP-ALIF, Izhikevich}
    "v_fix": False,
    "v_fix_psi": False,
    "fraction_ALIF": 0.25,  # def 0.25
    "n_directions": 1,  # Reduces error from 36.1 to 32.9.
    "seed": None,  # 'None' for random seed

    "alpha": 0.78,  # .78 for 1, .95 for 5
    "rho": 0.975,  # .975 for 1, .995 for 5
    "kappa": 0.717,
    "beta": 0.184,  # Bellec2: "order of 0.07", Bellec3: 0.184. Code: 1.8
    "gamma": 0.3,  # Bellec2: 0.3
    "thr": 1.6,  # Bellec3: 1.6
    "dt_refr": 2,  # Bellec3: 2

    "dropout": 0,

    "eta_W": 0.01,  # for fast
    "eta_W_in": 0.01,
    "eta_W_rec": 0.01,
    "eta_out": 0.01,
    "eta_bias": 0.01,
    "eta_decay": 0,

    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-5,

    "weight_decay": 1e-2,  # Bellec3: 1e-2
    "L2_reg": 0,  # Bellec3: 1e-5
    "FR_target": 0.01,  # BellecCode: 0.01 (10hz)
    "FR_reg": 50,  # Bellec3: 50

    "batch_op": 'mean',
    "uniform_dist": False,
    "weightscale": 1,

    "N_R": 200,
    "N_Rec": 1,

    "task": "TIMIT_small",
    "wavs_fname": "../data/data_wavs",
    "phns_fname": "../data/data_phns",

    "cuda": True,

    "warmup": False,
    "one_to_one_output": False,

    "train_W": True,  # for fast
    "train_W_in": True,
    "train_W_rec": True,
    "train_out": True,
    "train_bias": True,

    "Epochs": 500,  # def = 80
    "Track_neuron": True,
    "Track_synapse": False,
    "Repeats": 1,  # ms per epoch, def = 5
    "Interpolation": 'nearest',  # nearest, linear
    "batch_size_train": 12,  # def = 32
    "batch_size_val": 12,  # def = 32
    "batch_size_test": 2,  # def = 32
    "maxlen": 778,
    "TIMIT_derivative": 2,
    "n_examples": {'train': 100, 'val': 100, 'test': 50},  # Re-process TIMIT!
    # "n_examples": {'train': 3696, 'val': 400, 'test': 192},
    "plot_model_interval": 20,  # Per iter  #  State plot; 0 to disable plots
    "plot_tracker_interval": 10,  # Per it
    "state_save_interval": 50,
    "val_every_B": 10,
    "max_v_batches": 1,
    "visualize_val": False,

    "frame_size": 0.025,
    "frame_stride": 0.01,
    "pre_emphasis": 0.97,
    "nfilt": 26,
    "NFFT": 512,
    "num_ceps": 13,
    "cep_lifter": 22,
}

# cfg["rho"] = exp(-1/(cfg["rho_N"]))  # Bellec1: 200 = 0.995. 40: 0.975
# cfg["alpha"] = exp(-1/(cfg["alpha_N"]))  # Bellec1: 20 = 0.951. 4: 0.779
# cfg["kappa"] = exp(-1/(cfg["kappa_N"]))  # Bellec1: 3 = 0.717. .75:~0.25
# cfg["beta"] = cfg["beta"] *  (1 - exp(-1 / cfg["rho_N"])) / (1 - exp(-1 / cfg["alpha_N"]))


lookup = {
    "x":       {"binary":False, "label": "x"},
    "I":       {"binary":False, "label": "I"},
    "I_in":    {"binary":False, "label": "I_{{in}}"},
    "I_rec":   {"binary":False, "label": "I_{{rec}}"},
    "v":       {"binary":False, "label": "v"},
    "a":       {"binary":False, "label": "a"},
    "h":       {"binary":False, "label": "\\psi"},
    "z":       {"binary":True, "label": "z"},
    "z_in":    {"binary":True, "label": "z_{{in}}"},
    "y":       {"binary":False, "label": "y"},
    "t":       {"binary":True, "label": "\\pi^*"},
    "p":       {"binary":False, "label": "\\pi"},
    "d":       {"binary":True, "label": "D"},
    "pm":      {"binary":True, "label": "\\pi_{{max}}"},
    "correct": {"binary":False, "label": "Correct"},
    "vv":      {"binary":False, "label": "\\epsilon_v"},
    "va":      {"binary":False, "label": "\\epsilon_a"},
    "etbar":   {"binary":False, "label": "\\bar{{e}}"},
    "et":      {"binary":False, "label": "e"},
    "loss":    {"binary":False, "label": "L"},
    "loss_reg": {"binary":False, "label": "L_{{reg}}"},
    "loss_pred": {"binary":False, "label": "L_{{pred}}"},
    "GW_in": {"binary":False, "label": "G_{{in}}"},
    "GW_rec": {"binary":False, "label": "G_{{rec}}"},
    "Gout": {"binary":False, "label": "G_{{out}}"},
    "Gbias": {"binary":False, "label": "G_{{bias}}"},
}

if cfg["Repeats"] == 1:
    cfg["Interpolation"] = 'nearest'
