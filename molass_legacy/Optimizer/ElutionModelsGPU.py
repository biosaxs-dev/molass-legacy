# coding: utf-8
"""

    Optimize.ModelParams.py

    Copyright (c) 2017-2021, SAXS Team, KEK-PF

"""
import cupy as np

def gaussian(x, h, m, s):
    return h * np.exp(-((x-m)/s)**2)

def egh(x, H=1, tR=0, sigma=1.0, tau=0):
    x_  = x - tR
    s2  = 2 * sigma**2
    z   = s2 + tau*x_
    z_neg   = z <= 0
    z_pos   = z > 0

    zero_part = np.zeros( len(x) )[z_neg]
    posi_part = H * np.exp( -x_[z_pos]**2/( s2 + tau*x_[z_pos] ) )

    if tau > 0:
        parts = [ zero_part, posi_part ]
    else:
        parts = [ posi_part, zero_part ]

    return np.hstack( parts )
