import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import delta
from config import cfg

"""
N audioframes --> N class labels

Train set contains 4620 audio files
Test set contains 1680

Split train into 2940 train and 1680 val
also have 1680 test.

longest audio = 124621

"""
n_train = 3696
n_val = 400
n_test = 192
sampling_rate = 16000
windur = 25
stepdur = 10
winlen = sampling_rate * windur / 1000
steplen = sampling_rate * stepdur / 1000


def read_phn(fname):
    """Returns as [(start, end, phone)] tuple list."""
    with open(fname) as f:
        read_data = f.read()
    return read_dat

def read_sound(fname):
    """Returns as [(start, end, soundarr)] tuple list."""
    sig, rate = sf.read(fname)
    assert rate == sampling_rate
    mfcc_feat = mfcc(signal=sig, samplerate=rate)
    delta1 = delta(feat=mfcc_feat, N=2)
    delta2 = delta(feat=delta1, N=2)
    conc = np.concatenate((mfcc_feat, delta1, delta2), axis=1)
    return conc

maxnframes = 778
nwavs = cfg["TIMIT_ntrain"] + cfg["TIMIT_nval"]
wavdata = np.zeros(shape=(nwavs, maxnframes, 39))
phonedata = np.zeros(shape=(nwavs, maxnframes, 61))
idx = 0

known_phones = []
ohv_len = 61



def read_phonelines(lines):
    def to_ohv(p):
        try:
            idx = known_phones.index(p)
        except ValueError:
            known_phones.append(p)
            idx = len(known_phones) - 1
        ret = [0] * ohv_len
        try:
            ret[idx] = 1
        except IndexError:
            print("Error: increase ohv capacity. Data is now corrupted")

        return ret
    phones = [(to_ohv(p[:-1].rpartition(' ')[-1]), int(p.split(' ')[1])) 
                      for p in phonelines]
    return phones

def align_phones(phones):
    # For each frame, see at which point the average time of the frame fits
    # in the phone time scheme. 
    ret = np.zeros(shape=(maxnframes, ohv_len))
    prev_phonetime = -1
    phone_idx = 0
    currphone = phones[phone_idx][0]

    for frame in range(maxnframes):
        frametime = int(winlen / 2 + frame * steplen)
        for phone, phonetime in phones:
            if frametime < phonetime:
                ret[frame, :] = phone
                break

        else:
            break
    return ret



for subdir, dirs, files in os.walk('../TIMIT/TRAIN'):
    if idx == nwavs:
        break
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".PHN"):
            with open(filepath) as f:
                phonelines = f.readlines()
            phones = read_phonelines(phonelines)
            aligned = align_phones(phones)
            phonedata[idx, :, :] = aligned

        if filepath.endswith(".WAV"):
            sig, _ = sf.read(filepath)
            c = read_sound(filepath)

            wavdata[idx, :c.shape[0], :] = c

            idx += 1
            if idx == nwavs:
                break
            print(f"{idx}/{nwavs}")

print(wavdata.shape)
print(phonedata.shape)
np.save("../data_wavs", wavdata)
np.save("../data_phns", phonedata)
# plt.plot(mfcc_feat[:, :])
# plt.show()