"""
    CharFunc/PdfCfStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

import numpy as np
import matplotlib.pyplot as plt

def study_egh():
    from molass_legacy.Models.EGH import egh
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
    ax1.set_title("PDF Domain")
    ax2.set_title("CF Real Domain")
    ax3.set_title("CF Imaginary Domain")
    x = np.arange(300)
    y1 = egh(x, 1, 120, 20, 0)
    y2 = egh(x, 1, 150, 30, 10)
    ty = y1 + y2
    tz = np.fft.fft(ty)
    w = np.fft.fftfreq(len(ty))
    u = np.fft.ifft(tz)
    z1 = np.fft.fft(y1)
    z2 = np.fft.fft(y2)
    ax1.plot(x, ty, label="data")
    ax1.plot(x, y1, ":", label="component-1")
    ax1.plot(x, y2, ":", label="component-2")

    # ax2.plot(w, np.real(tz), "o", markersize=1)
    # ax3.plot(w, np.imag(tz), "o", markersize=1)
    ax2.plot(w, np.real(z1), "o", markersize=1)
    ax2.plot(w, np.real(z2), "o", markersize=1)
    ax3.plot(w, np.imag(z1), "o", markersize=1)
    ax3.plot(w, np.imag(z2), "o", markersize=1)

    ax1.plot(x, u, ":", color="cyan", label="inverse FFT")
    v = np.fft.ifft(z1 + z2)
    ax1.plot(x, v, ":", color="red", label="inverse FFT of sum")

    ax1.legend()
    fig.tight_layout()
    plt.show()

def study_phi():
    from SecTheory.BasicModels import single_pore_pdf
    from SecTheory.SecCF import simple_phi, shifted_phi

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20,4))
    ax1.set_title("PDF Domain")
    ax2.set_title("CF Real Domain")
    ax3.set_title("CF Imaginary Domain")

    x = np.arange(21, 321)
    x0 = 20
    npi = 100
    tpi = 1
    y1 = single_pore_pdf(x, npi, tpi)

    x = np.arange(21, 321)
    y1 = single_pore_pdf(x, npi, tpi)
    y2 = single_pore_pdf(x - x0, npi, tpi)
    ty = y1 + y2
    tz = np.fft.fft(ty)
    w = np.fft.fftfreq(len(ty))
    u = np.fft.ifft(tz)
    z1 = np.fft.fft(y1)
    z2 = np.fft.fft(y2)

    z1_ = simple_phi(w, npi, tpi)
    z2_ = shifted_phi(w, npi, tpi, -x0)
    u1_ = np.fft.ifft(z1_)
    u2_ = np.fft.ifft(z2_)

    ax1.plot(x, ty, label="data")
    ax1.plot(x, y1, ":", label="component-1")
    ax1.plot(x, y2, ":", label="component-2")

    ax1.plot(x, u1_, color="red", alpha=0.5, label="component-1 inverse FFT")
    ax1.plot(x, u2_, color="green", alpha=0.5, label="component-2 inverse FFT")

    # ax2.plot(w, np.real(tz), "o", markersize=1)
    # ax3.plot(w, np.imag(tz), "o", markersize=1)
    ax2.plot(w, np.real(z1), "o", markersize=1)
    ax2.plot(w, np.real(z2), "o", markersize=1)

    ax3.plot(w, np.imag(z1), "o", markersize=1)
    ax3.plot(w, np.imag(z2), "o", markersize=1)

    ax1.plot(x, u, ":", color="cyan", label="inverse FFT")
    v = np.fft.ifft(z1 + z2)
    ax1.plot(x, v, ":", color="red", label="inverse FFT of sum")

    ax1.legend()

    ax4.plot(w, "o", markersize=1)
    ax4.plot(0, w[0], "o", color="red")
    ax4.plot(len(w)-1, w[-1], "o", color="blue")

    fig.tight_layout()
    plt.show()

def study_phi_with_Witkovsky():
    from SecTheory.BasicModels import single_pore_pdf
    from SecTheory.SecCF import simple_phi, shifted_phi
    from SecTheory.SecPDF import BidirectFft

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20,4))
    ax1.set_title("PDF Domain")
    ax2.set_title("CF Real Domain")
    ax3.set_title("CF Imaginary Domain")

    x = np.arange(21, 321)
    x0 = 20
    npi = 100
    tpi = 1
    y1 = single_pore_pdf(x, npi, tpi)

    x = np.arange(21, 321)
    y1 = single_pore_pdf(x, npi, tpi)
    y2 = single_pore_pdf(x - x0, npi, tpi)
    ty = y1 + y2

    bifft = BidirectFft(x)

    tz = bifft.compute_z(ty)
    w = bifft.get_w()
    u = bifft.compute_y(tz)

    z1 = bifft.compute_z(y1)
    z1_ = simple_phi(w, npi, tpi)
    u1_ = bifft.compute_y(z1_)

    z2 = bifft.compute_z(y2)
    z2_ = shifted_phi(w, npi, tpi, x0)
    u2_ = bifft.compute_y(z2_)

    ax1.plot(x, ty, label="data")
    ax1.plot(x, y1, ":", label="component-1")
    ax1.plot(x, y2, ":", label="component-2")

    ax1.plot(x, u1_, color="red", alpha=0.5, label="component-1 inverse FFT")
    ax1.plot(x, u2_, color="green", alpha=0.5, label="component-2 inverse FFT")

    # ax2.plot(w, np.real(tz), "o", markersize=1)
    # ax3.plot(w, np.imag(tz), "o", markersize=1)
    ax2.plot(w, np.real(z1), "o", markersize=1)
    ax2.plot(w, np.real(z2), "o", markersize=1)

    ax3.plot(w, np.imag(z1), "o", markersize=1)
    ax3.plot(w, np.imag(z2), "o", markersize=1)

    ax1.plot(x, u, ":", color="cyan", label="inverse FFT")
    v = bifft.compute_y(z1 + z2)
    ax1.plot(x, v, ":", color="red", label="inverse FFT of sum")

    ax1.legend()

    ax4.plot(w, "o", markersize=1)
    ax4.plot(0, w[0], "o", color="red")
    ax4.plot(len(w)-1, w[-1], "o", color="blue")

    fig.tight_layout()
    plt.show()


def study_Gil_Pelaez():
    mu = 100
    sigma = 10
    def f(x):
        return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

    x = np.arange(300)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

    # study_egh()
    #study_phi()
    study_phi_with_Witkovsky()
    # study_Gil_Pelaez()