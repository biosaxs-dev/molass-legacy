"""
    SecTheory.BasicProofs.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.special import iv
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt

def simple_pdf(t, npi, tpi):
    return iv(1, np.sqrt(4*npi*t/tpi)) * np.sqrt(npi/(t*tpi)) * np.exp(-t/tpi-npi)

def numerical_inversion_proof(**kwargs):
    from .SecCF import simple_phi
    from .SecPDF import FftInvPdf

    fig, ax = plt.subplots()

    ax.set_title("Numerical Inversion Proof (1)", fontsize=16)

    x = np.arange(10, 300)
    npi = 20
    tpi = 5

    pdf = FftInvPdf(simple_phi)

    y1 = simple_pdf(x, npi, tpi)
    ax.plot(x, y1, label="formula pdf(t)")
    y2 = pdf(x, npi, tpi)
    ax.plot(x, y2, label="numerically inverted pdf(t)")
    ax.legend()

    print("sum(y1)=", np.sum(y1))
    print("sum(y2)=", np.sum(y2))

    fig.tight_layout()
    plt.show()

def roud_trip(in_folder=None):
    from scipy.optimize import curve_fit
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Peaks.ElutionModels import egh
    from LPM import get_corrected
    from DataUtils import get_in_folder
    from SecSaxs.DataTreatment import DataTreatment
    from .SecPDF import compute_standard_wCD

    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    pre_recog = PreliminaryRecognition(sd)
    treat = DataTreatment(route="v2", trimming=2, correction=1)
    sd_copy = treat.get_treated_sd(sd, pre_recog)

    D, E, qv, ecurve = sd_copy.get_xr_data_separate_ly()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20,5))
    fig.suptitle("Round Trip Proof for %s" % get_in_folder(in_folder), fontsize=20)

    ax1.set_title("Elution", fontsize=16)
    ax2.set_title("PDF", fontsize=16)
    ax3.set_title("CF(real)", fontsize=16)
    ax4.set_title("CF(imag)", fontsize=16)

    x = ecurve.x
    y = ecurve.y
    ax1.plot(x, y, label="data")

    area = np.sum(y)
    y_ = y/area
    ax2.plot(x, y_, label="pdf")

    N = 1024
    w, C, D = compute_standard_wCD(N)
    cft = []
    for w_ in w:
        cft.append(np.sum(np.exp(1j*w_*x)*y_))

    cft = np.array(cft)
    pdfFFT = np.max([np.zeros(N), (C*np.fft.fft(D*cft)).real], axis=0)
    spline = UnivariateSpline(np.arange(N), pdfFFT, s=0)
    inv_y = spline(x)
    ax2.plot(x, inv_y, label="inverted pdf")

    ax1.plot(x, inv_y*area, label="round tripped")

    ax3.plot(w, cft.real)
    ax4.plot(w, cft.imag)

    for ax in [ax1, ax2]:
        ax.legend()

    fig.tight_layout()
    plt.show()
