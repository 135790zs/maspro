import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import delta

"""
N audioframes --> N class labels

"""
n_train = 3696
n_val = 400
n_test = 192

def read_phn(fname):
    """Returns as [(start, end, phone)] tuple list."""
    with open(fname) as f:
        read_data = f.read()
    return read_dat

def read_sound(fname):
    """Returns as [(start, end, soundarr)] tuple list."""
    sig, rate = sf.read(fname)
    mfcc_feat = mfcc(signal=sig, samplerate=rate)
    delta1 = delta(feat=mfcc_feat, N=2)
    delta2 = delta(feat=delta1, N=2)
    conc = np.concatenate((mfcc_feat, delta1, delta2), axis=1)
    return conc


for subdir, dirs, files in os.walk('../TIMIT/TRAIN'):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".WAV"):
            print (filepath)

# plt.plot(mfcc_feat[:, :])
# plt.show()