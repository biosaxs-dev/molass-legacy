"""
    SecTheory.Sepsey2014b.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import basinhopping
import molass_legacy.KekLib.DebugPlot as plt

def ng(rg, rp, nperm, me):
    rho = rg/rp
    return nperm*(1 - rho)**me

def tg(rg, rp, tperm, mp):
    rho = rg/rp
    return tperm*(1 - rho)**mp

SQRT_2PI = np.sqrt(2*np.i)

def sepsey_phi_19(w, rp):
    pass

def demo():
    from .SecCF import simple_phi, moving_zone_phi
    from .SecPDF import c, FftInvPdf

    t = np.arange(300)

    npi = 10
    tpi = 10
    simple_pdf = FftInvPdf(simple_phi)

    moving_pdf = FftInvPdf(moving_zone_phi)
    N0 = 6
    t0 = 6

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("Proof of Numerical Inversion", fontsize=20)
        ax1.set_title("Simple", fontsize=16)
        ax1.plot(t, c(t, npi, tpi), label="formula")
        ax1.plot(t, simple_pdf(t, npi, tpi), label="numerical")
        ax1.legend()
        ax2.set_title("Moving Zone considered", fontsize=16)
        ax2.plot(t, moving_pdf(t, npi, tpi, N0, t0), label="numerical")
        ax2.legend()
        fig.tight_layout()
        plt.show()
