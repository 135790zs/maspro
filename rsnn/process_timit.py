import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import delta
from config import cfg


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
    mfcc_feat = mfcc(*sf.read(fname))
    delta1 = delta(feat=mfcc_feat, N=2)
    delta2 = delta(feat=delta1, N=2)
    conc = np.concatenate((mfcc_feat, delta1, delta2), axis=1)
    return conc


if __name__ == "__main__":

    sampling_rate = 16000
    windur = 25
    stepdur = 10
    winlen = sampling_rate * windur / 1000
    steplen = sampling_rate * stepdur / 1000

    maxnframes = cfg["maxlen"]  # def: 778

    known_phones = []
    ohv_len = 61

    for tvt_type in ['train', 'test']:
        idx = 0
        if tvt_type == 'test':
            nwavs = cfg["n_examples"]['test']
            print(f"\nReading {nwavs} test files...")
        elif tvt_type == 'train':
            nwavs = cfg["n_examples"]['train'] + cfg["n_examples"]['val']
            print(f"\nReading {nwavs} train & val files...")

        plotidx = np.random.default_rng().integers(nwavs)
        wavdata = np.zeros(shape=(nwavs, maxnframes, 39))
        phonedata = np.zeros(shape=(nwavs, maxnframes, 61))
        fig = plt.figure(constrained_layout=False, figsize=(14, 9))
        gsc = fig.add_gridspec(nrows=3, ncols=1, hspace=0.2)
        axs = [fig.add_subplot(gsc[r, :]) for r in range(3)]

        directory = tvt_type.upper()
        for subdir, dirs, files in os.walk(f'../TIMIT/{directory}'):
            if idx == nwavs:
                break
            for filename in files:
                filepath = subdir + os.sep + filename

                if filepath.endswith(".WAV"):
                    print(f"{idx+1}/{nwavs}")
                    sig, _ = sf.read(filepath)
                    c = read_sound(filepath)
                    wavdata[idx, :c.shape[0], :] = c[:maxnframes]
                    if idx == plotidx:
                        axs[0].plot(sig[:int(steplen*maxnframes)])
                        axs[0].margins(0)
                        axs[1].imshow(wavdata[idx, :c.shape[0]].T,
                                      interpolation='nearest',
                                      aspect='auto',
                                      cmap='coolwarm')

                    with open(f"{filepath[:-4]}.PHN") as f:
                        phonelines = f.readlines()
                    with open(f"{filepath[:-4]}.TXT") as f:
                        textline = f.read()
                    phones = read_phonelines(phonelines)
                    aligned, maxframe = align_phones(phones)
                    phonedata[idx, :, :] = aligned
                    if idx == plotidx:
                        plt.suptitle(' '.join(textline.split(' ')[2:]))
                        axs[2].imshow(aligned[:c.shape[0]].T,
                                      interpolation='nearest',
                                      aspect='auto',
                                      cmap='gray')
                        for line in phonelines:
                            start, end, phone = line.split(' ')
                            if int(start) >= maxnframes * steplen:
                                break
                            axs[0].axvline(int(start),
                                           color='black',
                                           label=phone)
                            axs[0].axis('off')
                            axs[0].text(x=(int(start) + int(end)) // 2,
                                        y=np.min(sig)*1.4,
                                        s=phone)


                    idx += 1
                    if idx == nwavs:
                        break

        if tvt_type == 'test':
            np.save(f'{cfg["wavs_fname"]}_{tvt_type}.npy', wavdata)
            np.save(f'{cfg["phns_fname"]}_{tvt_type}.npy', phonedata)
        elif tvt_type == 'train':
            np.save(f'{cfg["wavs_fname"]}_train.npy',
                    wavdata[:cfg['n_examples']['train']])
            np.save(f'{cfg["phns_fname"]}_train.npy',
                    phonedata[:cfg['n_examples']['train']])
            np.save(f'{cfg["wavs_fname"]}_val.npy',
                    wavdata[cfg['n_examples']['val']:])
            np.save(f'{cfg["phns_fname"]}_val.npy',
                    phonedata[cfg['n_examples']['val']:])
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(f"../vis/exampleTIMIT_{tvt_type}.pdf",
                    bbox_inches='tight')

        plt.close()
