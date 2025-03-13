"""
    Models.Stochastic.PaperFelinger2004.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def felinger_sdm_cf(w, n1, t1, Nd, t0):
    return np.exp(Nd * (1 - np.sqrt(1 - 2*1j*w/Nd*(n1*t1/(1 - 1j*t1*w) + t0))))

def felinger_sdm_cf_proof():
    from molass_legacy.Models.ElutionModelUtils import compute_4moments
    from molass_legacy.Models.RateTheory.EDM import guess_params_from_sdm
    from SecTheory.Edm import guess_single_edm, edm_func
    from SecTheory.SecCF import dispersive_monopore
    from SecTheory.SecPDF import FftInvPdf
    dispersive_monopore_pdf = FftInvPdf(dispersive_monopore)
    felinger_sdm_pdf_ = FftInvPdf(felinger_sdm_cf)

    x = np.arange(300)
    t0 = 50
    Nd = 14400
    n1 = 400
    t1 = 0.3

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    fig.suptitle("Numerical Proof of Equivalence among Dondi, Felinger SDM and Rehman EDM")
    ax1.set_title("Felinger-1999 and Dondi-2002 SDM Formulae")
    y_dondi = dispersive_monopore_pdf(x, n1, t1, Nd, t0)
    ax1.plot(x, y_dondi, label="Dondi-2002 SDM")
    ax1.plot(x, felinger_sdm_pdf_(x, n1, t1, Nd, t0), ":", label="Felinger-1999 SDM")
    ax1.legend()
    params = guess_params_from_sdm(x, y_dondi, n1, t1, Nd, t0)
    print("params=", params)
    # t0, u, a, b, e, Dz, cinj
    t0, u, a, b, e, Dz, cinj = params
    edm_y = edm_func(x-t0, u, a, b, e, Dz, cinj)

    ax2.set_title("Dondi-2002 SDM and Rehman-2021 EDM Formulae")
    ax2.plot(x, y_dondi, label="Dondi-2002 SDM")
    ax2.plot(x, edm_y, ":", label="Rehman-2021 EDM")

    ax2.legend()
    fig.tight_layout()
    plt.show()

def demo():
    from molass_legacy.Models.ElutionCurveModels import egh
    from SecTheory.Edm import guess_single_edm, edm_func

    # t, u, a, b, e, Dz, cinj
    x = np.arange(300)
    egh_y = egh(x, 1, 120, 20, 20)
    edm = guess_single_edm(x, egh_y)
    params = edm.get_comp_params()
    print("params=", params)
    u = 0.5
    edm_y = edm_func(x, u, *params)

    fig, ax = plt.subplots()
    ax.plot(x, egh_y, label="EGH")
    ax.plot(x, edm_y, ":",  label="EDM")
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    sys.path.append("../lib")

    felinger_sdm_cf_proof()
    # demo()