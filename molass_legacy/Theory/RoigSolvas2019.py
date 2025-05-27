"""
    Theory.RoigSolvas2019.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from .JsPedersen1997 import SolidEllipsoidRev, TriAxialEllipsoid

def ellipsoid_approx_intensities(q, Rg2, Af2):
    return 1 - Rg2/3 * q**2 + 1/21*(Rg2**2 + 1/5*Af2)*q**4

class Anisotropy:
    def __init__(self):
        pass

    def fit(self, q, Ic, initRg):
        initR = np.asrt(5/3)*initRg

    def fit_approx(self, q, Ic, initRg):

        def obj_func(p):
            Rg2, Af2 = p
            return np.sum((Ic - ellipsoid_approx_intensities(q, Rg2, Af2))**2)

        # method = 'SLSQP'
        # method = 'BFGS'
        # method = 'L-BFGS-B'
        # method = 'CG'
        initRg2 = initRg**2
        # res = minimize(obj_func, (initRg2, 0), method=method, bounds=((0, initRg2*2), (0, 1)))
        res = minimize(obj_func, (initRg2, 0))
        self.C = 1
        self.Rg2, self.Af2 = res.x
        return res.x

    def approx_intensities(self, q):
        return self.C * ellipsoid_approx_intensities(q, self.Rg2, self.Af2)

def demo(root, sd, pno=0, debug=True):
    from bisect import bisect_right
    from .SynthesizedLRF import synthesized_lrf_spike as synthesized_lrf
    from SvdDenoise import get_denoised_data
    from .Rg import compute_corrected_Rg, compute_Rg
    from .SolidSphere import get_boundary_params_simple, SolidSphere
    from .PbMoore1980 import ift_intensities
    from DataUtils import get_in_folder
    from .JsPedersen1997 import P6

    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[pno]
    f = range_[0]
    p = range_[1]
    t = range_[2]

    eslice = slice(f,t)
    x = ecurve.x
    y = ecurve.y

    M0 = M[:,eslice]

    c_ = y[eslice]
    M2 = get_denoised_data(M0, rank=2)
    C2 = np.array([c_, c_**2])
    P2 = M2 @ np.linalg.pinv(C2)

    Rg, gf, gt, _ = compute_corrected_Rg(sd, ecurve, pno, qv, M0, E[:,eslice])
    b1, b2, k = get_boundary_params_simple(Rg)
    P = synthesized_lrf(qv, M0, c_, M2, P2, boundary=b1, k=k)

    spt = P[:,0]
    # qc = np.hstack([[0], qv])
    qc = qv
    qc, Ic, Icerr, D = ift_intensities(qv, spt, E[:,p], qc=qc)
    scale = 1/Ic[0]
    spt_ = spt*scale
    Ic_ = Ic*scale

    sphere = SolidSphere()
    sphere.fit(qc, Ic_, Rg)
    y1 = sphere.intensity(qv)

    ellips = Anisotropy()
    i1 = bisect_right(qc, b1/2)
    ellips.fit_approx(qc[0:i1], Ic_[0:i1], Rg)
    y2 = ellips.approx_intensities(qv)
    print("Af2=%.2g" % ellips.Af2)
    af = np.sqrt(ellips.Af2)
    print("af=%.2g" % af)

    a, b, c = 3, 4, 5
    rg6 = np.sqrt((a**2 + b**2 + c**2)/5)
    scale = Rg/rg6

    p6 = P6(qc, a*scale, b*scale, c*scale)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,7))
    fig.suptitle("Anisotropy Factor Demo based on Roig-Solvas, 2019", fontsize=24)
    in_folder = get_in_folder()
    ax1.set_title(in_folder, fontsize=20)
    ax1.set_yscale('log')
    ax1.plot(qv, spt_, label='Syn LRF', color='C1')
    ax1.plot(qc, Ic_, label='Syn LRF IFT', color='C2')
    ax1.plot(qc, y1, label='Solid Sphere', color='yellow')
    ax1.plot(qc, y2, label='Fitting (Roig-Solvas, 2019)', color='cyan')

    ax1.text(0.15, 10e-2, "$A_F=%.2g$" % af, fontsize=50, alpha=0.3)

    ax1.legend(fontsize=16)

    i1 = bisect_right(qc, b1/2)
    ellips = Anisotropy()
    ellips.fit_approx(qc[0:i1], p6[0:i1], Rg)
    y2 = ellips.approx_intensities(qv)
    print("Af2=%.2g" % ellips.Af2)
    af = np.sqrt(ellips.Af2)
    print("af=%.2g" % af)

    ax2.set_yscale('log')
    ax2.set_title("Ellissoid(%.2g, %.2g, %.2g)" % (a*scale, b*scale, c*scale), fontsize=20)
    ax2.plot(qc, p6, label="Ellipsoid", color='green')
    ax2.plot(qc, y1, label='Solid Sphere', color='yellow')
    ax2.plot(qc, y2, label='Fitting (Roig-Solvas, 2019)', color='cyan')
    ax2.text(0.15, 10e-2, "$A_F=%.2g$" % af, fontsize=50, alpha=0.3)

    ax2.legend(fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()
