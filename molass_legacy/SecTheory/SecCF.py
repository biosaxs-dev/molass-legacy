"""
    SecTheory.SecCF.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np

def simple_phi(w, npi, tpi):
    return np.exp(npi*(1/(1 - 1j*w*tpi) - 1))

def shifted_phi(w, npi, tpi, x0):
    return np.exp(npi*(1/(1 - 1j*w*tpi) - 1) + 1j*w*x0)

def moving_zone_phi(w, npi, tpi, N0, x0):
    Z = npi*(1/(1 - 1j*w*tpi) - 1) + 1j*w*x0
    return np.exp(Z + 1/(2*N0) * Z**2)

def two_site_phi(w, n, p1, p2, t1, t2):
    # see 1999, Alberto Cavazzini
    # return np.exp(n*p1/(1 - 1j*w*t1) - n*p1) * np.exp(n*p2/(1 - 1j*w*t2) - n*p2)
    return np.exp(n*(p1/(1 - 1j*w*t1) + p2/(1 - 1j*w*t2) -1))

"""
    GEC (Giddings-Eyring-Carmichael) models of SEC
"""
def gec_monopore_phi(w, npi, tpi, x0):
    # same as shifted_phi above
    return np.exp(npi*(1/(1 - 1j*w*tpi) - 1) + 1j*w*x0)

def gec_dipore_phi(w, n1, t1, n2, t2, x0):
    return np.exp(n1*(1/(1 - 1j*w*t1) - 1) + n2*(1/(1 - 1j*w*t2) - 1) + 1j*w*x0)

def gec_tripore_phi(w, n1, t1, n2, t2, n3, t3, x0):
    return np.exp(n1*(1/(1 - 1j*w*t1) - 1) + n2*(1/(1 - 1j*w*t2) - 1)+ n3*(1/(1 - 1j*w*t3) - 1) + 1j*w*x0)

def gec_polypore_phi(w, nt_pairs, x0):
    """
        this does not work properly. why?
        z = np.sum([n_*(1/(1 - 1j*w*t_) - 1) for n_, t_ in nt_pairs])
    """
    z = 0
    for n_, t_ in nt_pairs:
        z += n_*(1/(1 - 1j*w*t_) - 1)
    return np.exp(z + 1j*w*x0)

def gec_oligopore_phi_impl(w, nt_pairs, props, t0):
    z = 0
    for (n_, t_), p in zip(nt_pairs, props):
        z += p*n_*(1/(1 - 1j*w*t_) - 1)
    return np.exp(z + 1j*w*t0)

def sec_oligopore_phi_impl(w, nt_pairs, props, t0, N0):
    # see 2002, F. Dondi, formula (75)
    z = 0
    for (n_, t_), p in zip(nt_pairs, props):
        u = n_*(1/(1 - 1j*w*t_) - 1) + 1j*w*t0 
        z += p * (u + 1/(2*N0) * u**2)          # be careful in proportional scaling. is this correct?
    return np.exp(z )

"""
    Stochastic-dispersive models of SEC
"""
def sdm_monopore(w, npi, tpi, N0, t0, x0):
    # same as moving_zone_phi above
    Z = npi*(1/(1 - 1j*w*tpi) - 1) + 1j*w*t0
    return np.exp(Z + 1/(2*N0) * Z**2 + 1j*w*x0)

def sdm_dipore(w, np1, tp1, np2, tp2, N0, t0, x0):
    Z1 = np1*(1/(1 - 1j*w*tp1) - 1) + 1j*w*t0
    Z2 = np2*(1/(1 - 1j*w*tp2) - 1) + 1j*w*t0
    return np.exp((Z1 + 1/(2*N0) * Z1**2) + (Z2 + 1/(2*N0) * Z2**2) + 1j*w*x0)

def sdm_tripore(w, np1, tp1, np2, tp2, np3, tp3, N0, t0, x0):
    Z1 = np1*(1/(1 - 1j*w*tp1) - 1) + 1j*w*t0
    Z2 = np2*(1/(1 - 1j*w*tp2) - 1) + 1j*w*t0
    Z3 = np3*(1/(1 - 1j*w*tp3) - 1) + 1j*w*t0
    return np.exp((Z1 + 1/(2*N0) * Z1**2) + (Z2 + 1/(2*N0) * Z2**2) + (Z3 + 1/(2*N0) * Z3**2) + 1j*w*x0)

"""
    Stochastic-dispersive models of SEC
"""
def dispersive_monopore(w, npi, tpi, N0, t0):
    Z = npi*(1/(1 - 1j*w*tpi) - 1) + 1j*w*t0
    return np.exp(Z + Z**2/(2*N0))
