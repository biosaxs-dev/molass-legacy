"""
    SimTools.ErrorModel.py - revised version of Error.ErrorModel.py 

    based on 2017, Steffen M. Sedlak, et al.
    Quantitative evaluation of statistical errors in smallangle X-ray scattering measurements

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import UnivariateSpline
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Peaks.ElutionModels import egh

def error_model(q, Iq, k, c, Iarb):
    return 1/(k*q)*(Iq + 2*c*Iarb/(1 - c))      # Eq.(10) in the paper

K_SCALE = 1e6

class ErrorModel:
    def __init__(self, qv, D, E, spline=None):
        self.E = E

        i = bisect_right(qv, 0.02)
        cy = np.average(D[i-5:i+6,:], axis=0)
        j = np.argmax(cy)
        sy = np.average(D[:,j-5:j+6], axis=1)
        se = np.average(E[:,j-5:j+6], axis=1)

        slice_ = slice(10, -10)
        # slice_ = slice(None, None)
        qv_ = qv[slice_]
        sy_ = sy[slice_]
        se_ = se[slice_]
        se2 = se_**2

        def obj_func(p):
            k, c = p
            y_ = error_model(qv_, sy_, k*K_SCALE, c, sy[i])
            return np.sum(((y_ - se2)/y_)**2)

        ret = minimize(obj_func, (1, 0.5), method="Nelder-Mead")
        k_, c_ = ret.x
        print("k_, c_=", k_, c_)
        print("ret.success=", ret.success)
        print("ret.nit=", ret.nit)

        sem =  error_model(qv_, sy_, k_*K_SCALE, c_, sy[i])

        if False:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_yscale("log")
                ax.plot(qv_, se_)
                ax.plot(qv_, np.sqrt(sem))
                fig.tight_layout()
                plt.show()

        if spline is None:
            spline = UnivariateSpline(qv, sy, s=0, ext=3)

        self.spline = spline
        self.Iarb = sy[i]
        self.params = ret.x

    def get_error_curve(self, qv, scale=1):
        k, c = self.params
        return np.sqrt(error_model(qv, scale*self.spline(qv), k*K_SCALE, c, scale*self.Iarb))

def demo(in_folder, v2_copy):
    from molass_legacy.DENSS.DenssUtils import fit_data_bc
    from DataUtils import get_in_folder
    from MatrixData import simple_plot_3d
    from molass_legacy.KekLib.BasicUtils import Struct
    from SvdDenoise import get_denoised_data

    D, E, qv, ecurve = v2_copy.get_xr_data_separate_ly()

    i = bisect_right(qv, 0.02)

    cx = np.arange(D.shape[1])
    cy = np.average(D[i-5:i+6,:], axis=0)

    p_init = ecurve.get_emg_peaks()[ecurve.primary_peak_no].get_params()
    popt, pcov = curve_fit(egh, cx, cy, p_init)
    cym = egh(cx, *popt)

    C = np.array([cym, cym**2])
    D_ = get_denoised_data(D, rank=2)
    P = D_ @ np.linalg.pinv(C)

    k = np.argmax(cym)

    aq, bq = P.T
    aq_ = aq*cym[k]

    j = np.argmax(cy)
    sy = np.average(D[:,j-5:j+6], axis=1)
    ey = np.average(E[:,j-5:j+6], axis=1)

    fq, fy, fe, _ = fit_data_bc(qv, sy, ey)
    spline = UnivariateSpline(fq, fy, s=0, ext=3)

    fq_, fy_, fe_, _ = fit_data_bc(qv, aq_, ey)

    em = ErrorModel(qv, D, E, spline=spline)
    eym = em.get_error_curve(qv)

    in_folder = get_in_folder(in_folder)
    super_title = "Error Model for %s based on 2017, Steffen M. Sedlak, et al." % in_folder

    def demo_plot():

        with plt.Dp():
            # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
            # ax1, ax2, ax3 = axes[0,:]
            fig = plt.figure(figsize=(15,10))
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            ax3 = fig.add_subplot(233)
            ax4 = fig.add_subplot(234, projection="3d")
            ax5 = fig.add_subplot(235, projection="3d")
            ax6 = fig.add_subplot(236, projection="3d")

            fig.suptitle(super_title, fontsize=20)
            ax1.set_title("Elution at Q=%g" % 0.02, fontsize=16)
            ax2.set_title("Scattering Intensity at j=%d" % j, fontsize=16)
            ax3.set_title("Scattering Error at j=%d" % j, fontsize=16)

            for ax in [ax2, ax3]:
                ax.set_yscale("log")
            ax1.plot(cx, cy, label="data")
            ax1.plot(cx, cym, label="egh")
            ax1.legend()

            ax2.plot(qv, sy, label="data")
            ax2.plot(fq, fy, ":", lw=2, label="IFT (P. B. Moore 1980) from data")
            ax2.plot(qv, aq_, label="A(q)")
            ax2.plot(fq, fy_, ":", lw=2, label="IFT (P. B. Moore 1980) from A(q)")
            ax2.legend()

            ax3.plot(qv, ey, label="data")
            ax3.plot(qv, eym, label="model")
            ax3.legend()

            ax4.set_zscale("log")
            simple_plot_3d(ax4, E, x=qv)

            fig.tight_layout()
            plt.show()

    demo_plot()

    from importlib import reload

    import Trials.BoundedLRF.IterativeLrfSolver
    reload(Trials.BoundedLRF.IterativeLrfSolver)

    from Trials.BoundedLRF.IterativeLrfSolver import IterativeLrfSolver

    f = 130
    t = 200
    range_ = slice(f, t+1)

    solver = IterativeLrfSolver(qv, D[:,range_], E[:,range_])
    P, C, Rg, R, L, bq_bounds, coerced_bq, changes = solver.solve(maxiter=100000, L=0.4)

    old_aq = aq_

    aq, bq = P.T
    aq_ = aq*cym[k]
    fq_, fy_, fe_, _ = fit_data_bc(qv, aq_, ey)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.plot(qv, sy, label="data")
        ax.plot(qv, old_aq, label="old A(q)")
        ax.plot(qv, aq_, label="new A(q)")
        ax.legend()
        fig.tight_layout()
        plt.show()

    super_title = "Result of Bounded LRF for %s" % in_folder
    demo_plot()
