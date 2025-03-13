"""
    SecTheory.MolassProof.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import basinhopping
from datetime import datetime
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from .SecCF import simple_phi, shifted_phi, moving_zone_phi, two_site_phi
from .SecPDF import FftInvPdf, c, FftInvImpl

def proof_1():

    t = np.arange(300)
    npi = 100
    tpi = 1
    simple_pdf = FftInvPdf(simple_phi)

    t0 = 20
    shifted_pdf = FftInvPdf(shifted_phi)

    moving_pdf = FftInvPdf(moving_zone_phi)
    # N0 = (t0/3)**2
    N0 = 5000

    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))

        ax1.plot(t, c(t, npi, tpi), label="formula")
        ax1.plot(t, simple_pdf(t, npi, tpi), label="numerical")

        finv = FftInvImpl()
        w = finv.get_w()
        z = simple_phi(w, npi, tpi)
        u = finv.compute(t, z)
        ax1.plot(t, u, ":", label="inverse FFT")

        if False:
            w = np.fft.fftfreq(len(t))
            z = simple_phi(w, npi, tpi)
            u = np.fft.ifft(z)
            ax1.plot(t, u, ":", label="inverse np.fft")

        ax1.legend()

        ax2.plot(t, c(t - t0, npi, tpi), label="shifted formula")
        ax2.plot(t, shifted_pdf(t, npi, tpi, t0), label="shifted numerical")
        ax2.legend()

        ax3.plot(t, c(t - t0, npi, tpi), label="shifted formula")
        ax3.plot(t, moving_pdf(t, npi, tpi, N0, t0), label="modified numerical")
        ax3.legend()

        fig.tight_layout()
        plt.show()

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
