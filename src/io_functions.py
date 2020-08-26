import numpy as np


def istream(time, amp=.1, freq=0.03, off_x=0, off_y=.2):
    return np.asarray([amp * np.sin(freq * (time + off_x)) + off_y])


def tstream(time, amp=.1, freq=0.06, off_x=0, off_y=.2):
    return np.asarray([amp * np.sin(freq * (time + off_x)) + off_y])
