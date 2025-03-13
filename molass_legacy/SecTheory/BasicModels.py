"""
    SecTheory.BasicModels.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.special import iv, ive
from scipy.interpolate import UnivariateSpline

def single_pore_pdf(t, np_, tp_):
    return iv(1, np.sqrt(4*np_*t/tp_)) * np.sqrt(np_/(t*tp_)) * np.exp(-t/tp_-np_)

SMALL_POS_VALUE = 1e-8

def robust_single_pore_pdf(t, np_, tp_, debug=False):
    # Bessel functions in Python that work with large exponents
    # https://stackoverflow.com/questions/13726464/bessel-functions-in-python-that-work-with-large-exponents
    #
    # iv(1, np.sqrt(4*np_*t/tp_)) * np.sqrt(np_/(t*tp_)) * np.exp(-t/tp_-np_)
    #
    # ive(v, z) = iv(v, z) * exp(-abs(z.real))
    # iv(v, sq) = ive(v, sq) * exp(sq)

    # val = single_pore_pdf(t, np_, tp_)
    sq = np.sqrt(4*np_*t/tp_)
    val = ive(1, sq) * np.sqrt(np_/(t*tp_)) * np.exp(sq -t/tp_ -np_)
    isnan_val = np.isnan(val)
    if debug:
        import molass_legacy.KekLib.DebugPlot as plt

        wh = np.where(isnan_val)[0]
        if len(wh) > 0:
            i, j = wh[[0, -1]]
            if i > 0 and val[i-1] > SMALL_POS_VALUE and j < len(val) - 1 and val[j+1] > SMALL_POS_VALUE:
                print("np_=", np_)
                print("tp_=", tp_)
                print("wh=", wh)
                if j < len(val) - 1:
                    print("val[j+1]=", val[j+1])
                isfinite = np.isfinite(val)
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.plot(t, val)
                    ax.plot(t[isnan_val], np.zeros(len(wh)), color="red")
                    fig.tight_layout()
                    plt.show()
    val[isnan_val] = 0
    return val

def fit_single_pore(x, y, moments=None):
    from scipy.optimize import minimize

    def f(p):
        h, t0, np_, tp_ = p
        y_ = h * robust_single_pore_pdf(x - t0, np_, tp_)
        return np.sum((y_ - y)**2)

    t0 = 0
    np_ = np.mean(x) if moments is None else moments[1]
    tp_ = 1
    ret = minimize(f, (0.1, t0, np_, tp_))
    return ret.x

def single_pore_height(np_, tp_):
    return ive(1, 2*np_) * np.sqrt(1/tp_**2)

def robust_single_pore_pdf_scaled(t, np_, tp_):
    return robust_single_pore_pdf(t, np_, tp_)/single_pore_height(np_, tp_)

def single_pore_elution(x, rg, rp, nperm, tperm, me, mp):
    if rg > rp:
        np_ = 0
        tp_ = 0
    else:
        rho = rg/rp
        np_ = nperm*(1 - rho)**me
        tp_ = tperm*(1 - rho)**mp
    return robust_single_pore_pdf(x, np_, tp_)

class SimpleSec:
    def __init__(self, rp0=124.34, nperm=2000, tperm=1):
        self.rp0 = rp0
        self.nperm = nperm
        self.tperm = tperm

    def get_elution(self, x, rg):
        return single_pore_elution(x, rg, self.rp0, self.nperm, self.tperm, 1, 1)

if __name__ == '__main__':
    import os
    import sys
    lib_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(lib_dir)
    kek_lib = os.path.join(lib_dir, "KekLib")
    sys.path.append(kek_lib)
    import molass_legacy.KekLib.DebugPlot as plt

    t = 10 + np.arange(300)
    y = robust_single_pore_pdf(t, 150, 1, debug=True)
    with plt.Dp():
        fig, ax = plt.subplots()
        ax.plot(t, y)
        plt.show()
