"""
    RollingStats.py

    Copyright (c) 2020-2023, Masatsuyo Takahashi, KEK-PF
"""

import numpy        as np

"""
    from:
        https://stackoverflow.com/questions/27427618/how-can-i-simply-calculate-the-rolling-moving-variance-of-a-time-series-in-pytho
        https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
"""
def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=int)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=int)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
