import numpy as np


def istream(time, amp=.2, freq=0.01, off_x=0, off_y=.2):
    return np.asarray([amp * np.sin(freq * (time + off_x)) + off_y])
