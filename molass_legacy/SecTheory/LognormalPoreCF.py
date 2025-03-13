"""
    SecTheory.LognormalPoreCF.py

    Copyright (c) 2023, SAXS Team, KEK-PF

    from
        2014a, Annamária Sepsey
        Molecular theory of size exclusion chromatography for wide pore size distributions

"""
import numpy as np

INV_R2PI = 1/np.sqrt(2*np.pi)

def lognormal_pore_phi_impl(w, rgv, Np, Tp, me, mp, mu, sigma, x0):
    m = np.exp(mu + sigma**2/2)
    s = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))
    minrp = max(0, m - 2*s)
    maxrp = m + 5*s
