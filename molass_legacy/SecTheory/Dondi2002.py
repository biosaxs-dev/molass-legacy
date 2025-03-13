# coding: utf-8
"""
    SecTheory.Dondi2002.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np

def monopore_gec(w, np_, tp_):
    return np.exp(np_*(1/(1 - 1j*w*tp_) - 1))

def twopore_gec(w, npa, tpa, npb, tpb):
    return np.exp(npa*(1/(1 - 1j*w*tpa) - 1) + npb*(1/(1 - 1j*w*tpb) - 1))

def dispersive_sec(w, np_, tp_, N0, t0):
    Z = np_*(1/(1 - 1j*w*tp_) - 1) + 1j*w*t0
    return np.exp(Z + 1/(2*N0)*Z**2)
