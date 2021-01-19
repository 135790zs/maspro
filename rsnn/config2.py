from numpy import exp
cfg = {
    "eprop_type": "symmetric",  # in {global, random, symmetric, adaptive}
    "Optimizer": "Adam",
    "v_fix": False,
    "fraction_ALIF": 0.25,  # def 0.25
    "n_directions": 1,  # Reduces error from 36.1 to 32.9.
    "seed": None,  # 'None' for random seed

    "beta": 0.184,  # Bellec2: "order of 0.07", Bellec3: 0.184. Code: 1.8
    "gamma": 0.3,  # Bellec2: 0.3
    "thr": 1.6,  # Bellec3: 1.6
    "dt_refr": 2,  # Bellec3: 2

    "dropout": 0,

    "eta_W": 0.01,
    "eta_out": 0.01,
    "eta_bias": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-5,

    "weight_decay": 1e-2,  # Bellec3: 1e-2
    "L2_reg": 0,  # Bellec3: 1e-5
    "FR_target": 0.01,  # BellecCode: 0.01 (10hz)
    "FR_reg": 50,  # Bellec3: 50
    "div_over_time": False,

    "N_R": 800,
    "N_Rec": 1,

    "task": "TIMIT",
    "wavs_fname": "../data/data_wavs",
    "phns_fname": "../data/data_phns",

    "cuda": True,

    "warmup": False,

    "train_W": True,
    "train_out": True,
    "train_bias": True,


    "Epochs": 130,  # def = 80
    "Track_synapse": False,
    "Track_neuron": True,
    "Repeats": 1,  # ms per epoch, def = 5
    "Interpolation": 'linear',  # nearest, linear
    "batch_size_train": 18,  # def = 32
    "batch_size_val": 18,  # def = 32
    "batch_size_test": 32,  # def = 32
    "maxlen": 778,  #def 778, Don't forget to re-run process_timit.py!
    "TIMIT_derivative": 2,
    "n_examples": {'train': 128, 'val': 128, 'test': 39},
    # # "n_examples": {'train': 3696, 'val': 400, 'test': 192},
    "plot_model_interval": 1,  # Per iter  #  State plot; 0 to disable plots
    "plot_tracker_interval": 1,  # Per epoch
    "state_save_interval": 1000,
    "val_every_B": 10,

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
    "x":       {"binary":False, "label": "x"},
    "I":       {"binary":False, "label": "I"},
    "I_in":    {"binary":False, "label": "I_{{in}}"},
    "I_rec":   {"binary":False, "label": "I_{{rec}}"},
    "v":       {"binary":False, "label": "v"},
    "a":       {"binary":False, "label": "a"},
    "h":       {"binary":False, "label": "\\psi"},
    "z":       {"binary":True, "label": "z"},
    "l":       {"binary":False, "label": "L"},
    "l_fr":    {"binary":False, "label": "L_{{fr}}"},
    "l_std":   {"binary":False, "label": "L_{{std}}"},
    "y":       {"binary":False, "label": "y"},
    "t":       {"binary":True, "label": "\\pi^*"},
    "p":       {"binary":True, "label": "\\pi"},
    "d":       {"binary":True, "label": "D"},
    "pm":      {"binary":True, "label": "\\pi_{{max}}"},
    "correct": {"binary":False, "label": "Correct"},
    "vv":      {"binary":False, "label": "\\epsilon_v"},
    "va":      {"binary":False, "label": "\\epsilon_a"},
    "etbar":   {"binary":False, "label": "\\bar{{e}}"},
    "et":      {"binary":False, "label": "e"},
}

if cfg["Repeats"] == 1:
    cfg["Interpolation"] = 'nearest'
