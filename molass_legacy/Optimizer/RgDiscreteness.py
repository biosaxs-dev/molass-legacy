"""
    RgDiscreteness.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt

VERY_SMALL_VALUE = 1e-10

def compute_rg_discreteness(rgs, unit=1):
    d = max(VERY_SMALL_VALUE, np.min(np.abs(np.diff(rgs))))
    d_ = min(unit, d)
    return (unit/d_ - 1)**2

if __name__ == '__main__':
    from matplotlib import cm

    for rgs in [[70, 32, 30], [70, 31, 30], [70, 31, 30.5], [70, 31, 31]]:
        print(rgs, compute_rg_discreteness(rgs, unit=1.5))
