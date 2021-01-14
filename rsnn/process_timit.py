import os
import soundfile as sf
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from python_speech_features import delta
from config2 import cfg

def my_mfcc(signal, sample_rate):
    if not os.path.isfile("../vis/signal.pdf"):
        plt.plot(signal)
        plt.savefig("../vis/signal.pdf")
        plt.clf()

    emphasized_signal = np.append(signal[0],
                                  signal[1:] - cfg["pre_emphasis"] * signal[:-1])

    if not os.path.isfile("../vis/signalemph.pdf"):
        plt.plot(emphasized_signal)
        plt.savefig("../vis/signalemph.pdf")
        plt.clf()

    frame_length = cfg["frame_size"] * sample_rate
    frame_step = cfg["frame_stride"] * sample_rate  # Convert from seconds to samples

    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(
        float(np.abs(
            signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # frames *= np.hamming(frame_length)
    frames *= 0.53836 - 0.46164 * np.cos((2 * np.pi * frame_length) / (frame_length - 1))  # Explicit Implementation **

    mag_frames = np.absolute(np.fft.rfft(frames, cfg["NFFT"]))  # Magnitude of the FFT
    if not os.path.isfile("../vis/magframes.pdf"):
        plt.imshow(mag_frames, interpolation='none', cmap='jet')
        plt.xlabel("Frame")
        plt.colorbar()
        plt.savefig("../vis/magframes.pdf")
        plt.clf()

    pow_frames = ((1.0 / cfg["NFFT"]) * ((mag_frames) ** 2))  # Power Spectrum

    if not os.path.isfile("../vis/powframes.pdf"):
        plt.imshow(pow_frames, interpolation='none', cmap='jet')
        plt.xlabel("Frame")
        plt.colorbar()
        plt.savefig("../vis/powframes.pdf")
        plt.clf()

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, cfg["nfilt"] + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((cfg["NFFT"] + 1) * hz_points / sample_rate)

    fbank = np.zeros((cfg["nfilt"], int(np.floor(cfg["NFFT"] / 2 + 1))))
    for m in range(1, cfg["nfilt"] + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    if not os.path.isfile("../vis/fbanks.pdf"):
        for bank in fbank:
            plt.plot(bank, color='black')
        plt.savefig("../vis/fbanks.pdf")
        plt.clf()

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    if not os.path.isfile("../vis/spectrogram.pdf"):
        plt.imshow(filter_banks.T, interpolation='none', cmap='jet', aspect='auto')
        plt.savefig("../vis/spectrogram.pdf")
        plt.clf()

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (cfg["num_ceps"] + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cfg["cep_lifter"] / 2) * np.sin(np.pi * n / cfg["cep_lifter"])
    mfcc *= lift  #*

    if not os.path.isfile("../vis/mfcc.pdf"):
        plt.imshow(mfcc.T, interpolation='none', cmap='jet', aspect='auto')
        plt.savefig("../vis/mfcc.pdf")
        plt.clf()
    filter_banks -= np.mean(filter_banks, axis=0)

    if not os.path.isfile("../vis/norm_fbanks.pdf"):
        plt.imshow(filter_banks.T, interpolation='none', cmap='jet', aspect='auto')
        plt.savefig("../vis/norm_fbanks.pdf")
        plt.clf()

    mfcc -= np.mean(mfcc, axis=0)

    if not os.path.isfile("../vis/norm_mfcc.pdf"):
        plt.imshow(mfcc.T, interpolation='none', cmap='jet', aspect='auto')
        plt.savefig("../vis/norm_mfcc.pdf")
        plt.clf()
    return mfcc


def read_phonelines(phonelines_):
    def to_ohv(p):
        try:
            ohv_idx = known_phones.index(p)
        except ValueError:
            known_phones.append(p)
            ohv_idx = len(known_phones) - 1
        ret = [0] * ohv_len
        try:
            ret[ohv_idx] = 1
        except IndexError:
            print("Error: increase ohv capacity. Data is now corrupted")

        return ret
    return [(to_ohv(p[:-1].rpartition(' ')[-1]), int(p.split(' ')[1]))
            for p in phonelines_]


def align_phones(phones_):
    # For each frame, see at which point the average time of the frame fits
    # in the phone time scheme.
    ret = np.zeros(shape=(maxnframes, ohv_len))
    frame = 0
    for frame in range(maxnframes):
        frametime = int(winlen / 2 + frame * steplen)
        # print(frametime)
        for phone, phonetime in phones_:
            # print(phonetime)
            if frametime < phonetime:
                # A phone discovered after this frametime!
                ret[frame, :] = phone
                break

        else:
            # No phones discovered after frametime! Silent.
            ret[frame, :] = phones_[0][0]

    return ret, frame


def read_sound(fname):
    """Returns as [(start, end, soundarr)] tuple list."""
    mfcc_feat = my_mfcc(*sf.read(fname))
    if cfg["TIMIT_derivative"] == 0:
        return mfcc_feat
    delta1 = delta(feat=mfcc_feat, N=2)
    conc = np.concatenate((mfcc_feat, delta1), axis=1)

    if cfg["TIMIT_derivative"] == 1:
        return conc

    delta2 = delta(feat=delta1, N=2)
    conc = np.concatenate((conc, delta2), axis=1)

    return conc


if __name__ == "__main__":

    sampling_rate = 16000
    ohv_len = 61

    nfeat = cfg["num_ceps"] * (cfg["TIMIT_derivative"] + 1)
    winlen = sampling_rate * cfg["frame_size"]
    steplen = sampling_rate * cfg["frame_stride"]

    maxnframes = cfg["maxlen"]  # def: 778

    known_phones = []

    for tvt_type in ['train', 'test']:
        idx = 0
        if tvt_type == 'test':
            nwavs = cfg["n_examples"]['test']
            print(f"\nReading {nwavs} test files...")
        elif tvt_type == 'train':
            nwavs = cfg["n_examples"]['train'] + cfg["n_examples"]['val']
            print(f"\nReading {nwavs} train & val files...")

        wavdata = np.ones(shape=(nwavs, maxnframes, nfeat)) * -1  # -1 used to truncate padding in code
        phonedata = np.zeros(shape=(nwavs, maxnframes, 61))

        directory = tvt_type.upper()
        for subdir, dirs, files in os.walk(f'../TIMIT/{directory}'):
            if idx == nwavs:
                break
            for filename in files:
                filepath = subdir + os.sep + filename

                if filepath.endswith(".WAV"):
                    print(f"{idx+1}/{nwavs}", end='\r')
                    sig, _ = sf.read(filepath)
                    c = read_sound(filepath)
                    wavdata[idx, :c.shape[0], :] = c[:maxnframes]

                    with open(f"{filepath[:-4]}.PHN") as f:
                        phonelines = f.readlines()
                    with open(f"{filepath[:-4]}.TXT") as f:
                        textline = f.read()
                    phones = read_phonelines(phonelines)
                    aligned, maxframe = align_phones(phones)
                    phonedata[idx, :, :] = aligned

                    idx += 1
                    if idx == nwavs:
                        break

        d_ext = "_small" if cfg["n_examples"]['train'] < 3696 else ""

        if tvt_type == 'test':
            np.save(f'{cfg["wavs_fname"]}_{tvt_type}_TIMIT{d_ext}.npy', wavdata)
            np.save(f'{cfg["phns_fname"]}_{tvt_type}_TIMIT{d_ext}.npy', phonedata)
        elif tvt_type == 'train':

            np.save(f'{cfg["wavs_fname"]}_train_TIMIT{d_ext}.npy',
                    wavdata[:cfg['n_examples']['train']])
            np.save(f'{cfg["wavs_fname"]}_val_TIMIT{d_ext}.npy',
                    wavdata[cfg['n_examples']['train']:])

            np.save(f'{cfg["phns_fname"]}_train_TIMIT{d_ext}.npy',
                    phonedata[:cfg['n_examples']['train']])
            np.save(f'{cfg["phns_fname"]}_val_TIMIT{d_ext}.npy',
                    phonedata[cfg['n_examples']['train']:])

    print("\nDone parsing TIMIT dataset!")
