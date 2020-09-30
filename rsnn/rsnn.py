"""
Construct the network

"""

from config import config
import io_fns
import numpy as np
import utils as ut

rng = np.random.default_rng()

x = np.zeros(size=(config['N_I']))
s = np.zeros(size=(config['N_rec'], config['N_R'], 2))
ev = np.zeros(size=(config['N_rec'], config['N_R'], 2))
W_rec = rng.random(size=(config['N_rec'], config['N_R'], config['N_R']))
W_ffd = rng.random(size=(config['N_rec']-1, config['N_R'], config['N_R']))
W_in = rng.random(size=(config['N_I'], config['N_R']))
W_out = rng.random(size=(config['N_R'], config['N_O']))
B = rng.random(size=(config['N_rec'], config['N_R']))


t = 0

while t < config["N_E"]:
    x = io_fns.random_input()
    s[]



    t += 1
