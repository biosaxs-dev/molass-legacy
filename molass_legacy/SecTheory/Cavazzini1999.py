"""
    SecTheory.Cavazzini1999.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np

def single_site_cf(w, n, t):
    return np.exp(n*(1/(1 - 1j*w*t) - 1))

def two_site_cf(w, n, p1, t1, t2):
    p2 = 1 - p1
    return np.exp(n*(p1/(1 - 1j*w*t1) + p2/(1 - 1j*w*t2) - 1))

def three_site_cf(w, n, p1, p2, t1, t2, t3):
    p3 = 1 - p1 - p2
    return np.exp(n*(-1 + p1/(1 - 1j*w*t1) + p2/(1 - 1j*w*t2) + p3/(1 - 1j*w*t3)))

def uniform_adsorption_energy(w, n, t1, t2):
    return np.exp(n*(np.log((1 - 1j*w*t1)/(1 - 1j*w*t2))/np.log(t2/t1)))

def uniform_mean_sojourn_time(w, n, t1, t2):
    return np.exp(n*(1/(1j*w*(t2 - t1))*np.log((1 - 1j*w*t1)/(1 - 1j*w*t2)) - 1))
