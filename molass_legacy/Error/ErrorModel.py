# coding: utf-8
"""
    Error.ErrorModel.py

    based on 2017, Steffen M. Sedlak, et al.
    Quantitative evaluation of statistical errors in smallangle X-ray scattering measurements

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
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

def demo_impl(in_folder, v2_copy):
    from molass_legacy.DENSS.DenssUtils import fit_data_bc
    from DataUtils import get_in_folder
    from MatrixData import simple_plot_3d
    from molass_legacy.KekLib.BasicUtils import Struct

    D, E, qv, ecurve = v2_copy.get_xr_data_separate_ly()

    i = bisect_right(qv, 0.02)

    cx = np.arange(D.shape[1])
    cy = np.average(D[i-5:i+6,:], axis=0)

    p_init = ecurve.get_emg_peaks()[ecurve.primary_peak_no].get_params()
    popt, pcov = curve_fit(egh, cx, cy, p_init)
    cym = egh(cx, *popt)

    j = np.argmax(cy)
    sy = np.average(D[:,j-5:j+6], axis=1)
    ey = np.average(E[:,j-5:j+6], axis=1)

    fq, fy, fe, _ = fit_data_bc(qv, sy, ey)
    spline = UnivariateSpline(fq, fy, s=0, ext=3)

    em = ErrorModel(qv, D, E, spline=spline)
    eym = em.get_error_curve(qv)

    with plt.Dp():
        in_folder = get_in_folder(in_folder)
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))
        fig.suptitle("Error Model for %s based on 2017, Steffen M. Sedlak, et al." % in_folder, fontsize=20)
        ax1.set_title("Elution at Q=%g" % 0.02, fontsize=16)
        ax2.set_title("Scattering Intensity at j=%d" % j, fontsize=16)
        ax3.set_title("Scattering Error at j=%d" % j, fontsize=16)

        for ax in [ax2, ax3]:
            ax.set_yscale("log")
        ax1.plot(cx, cy, label="data")
        ax1.plot(cx, cym, label="egh")
        ax1.legend()

        ax2.plot(qv, sy, label="data")
        ax2.plot(fq, fy, label="P. B. Moore 1980")
        ax2.legend()

        ax3.plot(qv, ey, label="data")
        ax3.plot(qv, eym, label="model")
        ax3.legend()

        fig.tight_layout()
        plt.show()

    def demo2_impl():
        print("popt=", popt)
        h = popt[0]

        error_curves = []
        for k in cx:
            scale = cym[k]/h
            error_curves.append(em.get_error_curve(qv, scale=scale))

        ME = np.array(error_curves).T
        print(D.shape, ME.shape)

        if False:
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
                ax1.plot(cx, cym)
                ax2.plot(cx, ME[i,:])
                fig.tight_layout()
                plt.show()

        fym = spline(qv)
        P_list = []
        ratio = 1/fym[i]
        MD = ratio*fym[:,np.newaxis] @ cym[np.newaxis,:]
        print("MD.shape=", MD.shape)

        with plt.Dp():
            fig = plt.figure(figsize=(15,10))
            # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5), subplot_kw=dict(projection="3d"))
            gs = GridSpec(2, 3)
            ax1, ax2, ax3 = [fig.add_subplot(gs[0,j], projection="3d") for j in range(3)]
            ax4, ax5, ax6 = [fig.add_subplot(gs[1,j]) for j in range(3)]
            simple_plot_3d(ax1, D, x=qv)
            simple_plot_3d(ax2, MD, x=qv)
            simple_plot_3d(ax3, ME, x=qv)
            fig.tight_layout()
            plt.show()

        exit()

        Mp_list = []
        Ma_list = []
        P_list = []
        rank = 1
        for n in range(100):
            E_ = ME * np.random.normal(0, 1, D.shape)
            M = MD + E_

            Mp_list.append(M)

            if False:

                with plt.Dp():
                    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,4))
                    ax3.set_yscale("log")
                    ax1.plot(cx, cym, label="start model")
                    ax1.plot(cx, M[i,:], label="perturbated")
                    ax1.legend()
                    for ax in [ax2, ax3]:
                        ax.plot(qv, fym, label="start model")
                        ax.plot(qv, M[:,j], label="perturbated")
                        ax.legend()
                    fig.tight_layout()
                    plt.show()

            U, s, VT = np.linalg.svd(M)
            M_ = U[:,0:rank] @ np.diag(s[0:rank]) @ VT[0:rank,:]
            Ma_list.append(M_)

            c = M_[i,:]
            C = np.array([c])
            Cinv = np.linalg.pinv(C)
            P = M_ @ Cinv
            P_list.append(P[:,0])

        Mp_std = np.std(Mp_list, axis=0)
        Ma_std = np.std(Ma_list, axis=0)

        with plt.Dp():
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5), subplot_kw=dict(projection="3d"))
            simple_plot_3d(ax1, ME, x=qv)
            simple_plot_3d(ax2, Mp_std, x=qv)
            simple_plot_3d(ax3, Ma_std, x=qv)
            fig.tight_layout()
            plt.show()

        P_std = np.std(P_list, axis=0)
        P = P_list[0]
        """
        P = D @ X
        X = D+ @ P
        Pe = np.sqrt((E**2) @ (X**2))
        """
        Dinv = np.linalg.pinv(D)
        X = Dinv @ P[:,np.newaxis]
        Pe = np.sqrt((ME**2) @ (X**2))

        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_yscale("log")
            ax.plot(qv, ME[:,j]/sy[i], label="data")
            ax.plot(qv, Pe/P[i], label="fixed propagation")
            ax.plot(qv, P_std/P[i], label="Monte Carlo")
            ax.legend()
            fig.tight_layout()
            plt.show()

    return demo2_impl

def demo(in_folder, trimming=True, correction=True):
    from molass_legacy.Tools.SdUtils import get_sd

    sd = get_sd(in_folder, trimming=trimming, correction=correction)
    demo2_impl = demo_impl(in_folder, sd)
    demo2_impl()
