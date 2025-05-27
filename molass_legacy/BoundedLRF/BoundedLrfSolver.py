"""
    BoundedLrfSolver.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.LRF.LrfRgUtils import compute_rg_from_qvDEP
from molass_legacy.Theory.SolidSphere import phi
from .ErrorCorrection import compute_corrected_error

class BoundedLrfSolver:
    def __init__(self, qv, D, E, C=None, i=None, denoise=True, debug_info=None):
        if i is None:
            from bisect import bisect_right
            i = bisect_right(qv, 0.02)

        self.debug_info = debug_info
        self.qv = qv
        self.D = D
        self.i = i
        if C is None:
            cy = np.average(D_[max(0, i-5):i+6,:], axis=0)      # i=3 with 20200304_1
            self.Cinit = np.array([cy, cy**2])
        else:
            cy = C[0,:]
            self.Cinit = C
        self.j = np.argmax(cy)

        if denoise:
            D_ = get_denoised_data(D, rank=C.shape[0])
        else:
            D_ = D

        self.D_ = D_
        self.E = E

    def solve(self, debug=False):
        qv = self.qv
        i = self.i
        j = self.j
        C = self.Cinit
        D_ = self.D_
        try:
            Cinv = np.linalg.pinv(C)
        except:
            cx = np.arange(C.shape[1])
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("solve pinv debug")
                for k, cy in enumerate(C):
                    print([k], cy)
                    ax.plot(cx, cy)
                fig.tight_layout()
                plt.show()
        P = D_ @ Cinv
        c1, c2 = C[0:2,j]

        Rg, R, L_, hK, hL = self.estimate_RgL(P, c1, debug=debug)

        aq, bq = P.T[0:2]

        # P_ = P.copy()
        normD = np.linalg.norm(self.D_)
        normBq = np.linalg.norm(bq)
        P_ = P.copy()

        def coerce_bounds(L_, return_info=False):
            bound = 1/(qv*L_*R)**2

            bq_bound = bound * aq/c1
            bq_bounds = (-bq_bound, bq_bound)

            coerced_bq = P_[:,1]
            where = bq < bq_bounds[0]
            coerced_bq[where] = bq_bounds[0][where]
            where = bq > bq_bounds[1]
            coerced_bq[where] = bq_bounds[1][where]

            P_[:,0] = aq + (bq - coerced_bq)*c2/c1

            if return_info:
                return bq_bounds, coerced_bq, P_

            return np.log10(np.linalg.norm(self.D_ - P_ @ C)/normD) + np.log10(np.linalg.norm(coerced_bq - bq)/normBq)

        bq_bounds, coerced_bq, P_ = coerce_bounds(L_, return_info=True)

        debug_plot = debug and self.debug_info is not None
        if debug_plot:
            from DataUtils import get_in_folder
            print("L_=", L_)
            in_folder = get_in_folder()
            aq_, bq_ = P_.T
            with plt.Dp():
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
                fig.suptitle("Bounded LRF Debug using %s" % in_folder)
                ax2.set_yscale("log")
                for i, ax in enumerate((ax1, ax2)):
                    ax.plot(qv, aq*c1)
                    ax.plot(qv, aq_*c1)
                    if i == 0:
                        ax.plot(qv, bq*c2)
                        ax.plot(qv, bq_*c2)
                ax3.plot(qv, bq*c2)
                ax3.plot(qv, coerced_bq*c2)
                ax3.plot(qv, bq_bounds[0]*c2)
                ax3.plot(qv, bq_bounds[1]*c2)
                ax3.set_ylim(ax1.get_ylim())
                fig.tight_layout()
                plt.show()

        D_pinv = np.linalg.pinv(self.D)
        W   = np.dot(D_pinv, P_)
        Pe  = np.sqrt(np.dot(self.E**2, W**2) )

        self.Pe_, adjust_scale = compute_corrected_error(qv, i, P, c1, c2, P_, Pe, bq_bounds, coerced_bq)

        return P_, C, Rg, R, L_, hK, hL, bq_bounds, coerced_bq

    def get_corrected_error(self):
        return self.Pe_

    def estimate_RgL(self, P, c1, debug=False):
        from scipy.optimize import basinhopping
        qv = self.qv
        aq, bq = P.T[0:2]
        sg = compute_rg_from_qvDEP(qv, self.D_, self.E, P, return_sg=True)
        Rg = sg.Rg

        debug_plot = debug and self.debug_info is not None
        if False:
            if Rg is None:
                glim = 0.02
            else:
                glim = 1.5/Rg
            glim_i = bisect_right(qv, glim)
            gslice = slice(0,glim_i)
            print("gslice=", gslice)
            q = (1.3/Rg)**2
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("estimate_RgL: Guinier Plot $R_g=%.3g$" % Rg, fontsize=20)
                ax.set_ylabel(r"$log_e(I)$")
                ax.set_xlabel(r"$Q^2$")
                ax.plot(qv[gslice]**2, np.log(aq[gslice]))
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                ax.plot([q, q], [ymin, ymax], color="yellow", label=r"$qR_g=1.3$")
                ax.legend()
                fig.tight_layout()
                plt.show()

        R = np.sqrt(5/3)*Rg
        K = 0.3
        L = 1

        i1 = bisect_right(qv, 0.01)
        i2 = bisect_right(qv, 0.2)
        slice_ = slice(i1,i2)
        qv_ = qv[slice_]
        aq_ = aq[slice_]
        bq_ = bq[slice_]

        def objective(p):
            K_, L_ = p
            h_bq = -K_*phi(qv_, 2*L_*R)*aq_
            negetive_penalty = min(0, K_)**2 + min(0, L_ - 2)**2
            return np.mean((h_bq - bq_)**2) + negetive_penalty

        # ret = minimize(objective, (K, L), method="Nelder-Mead")
        ret = minimize(objective, (K, L))

        if debug_plot:
            print("c1=", c1)
            K, L = ret.x
            print("(1) K, L =", K, L)
            h_bq_ = -K*phi(qv_, 2*L*R)*aq_
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("estimate_RgL debug")
                ax.plot(qv_, bq_, label="Data B(q)")
                ax.plot(qv_, h_bq_, label="Sphere B(q)")
                ax.legend()
                fig.tight_layout()
                plt.show()

        ig = bisect_right(qv, 0.06)
        gslice = slice(0, ig)
        qvg = qv[gslice]
        qvg2 = qvg**2

        if debug_plot:
            from DataUtils import get_in_folder
            K = 0.29
            L = 1.9
            j = self.j
            Icq = self.D_[:,j]
            d_aq = aq/sg.Iz
            h_aq = phi(qv, R)**2

            Hq = h_aq * (1 - K*phi(qv, L*2*R))
            scale = np.max(Hq)/np.max(Icq)
            Iq = Icq * scale

            # h_aq_ = phi(qv_, R)**2
            h_bq = - h_aq * K * phi(qv, L*2*R)

            Iqg = Iq[gslice]
            Hqg = Hq[gslice]

            d_aqg = d_aq[gslice]
            h_aqg = h_aq[gslice]

            if False:

                def guinier_objective(p):
                    K_, L_ = p
                    Hqg_ = h_aqg * (1 - K_*phi(qvg, L_*2*R))
                    return np.sum((np.log(Hqg_) - np.log(Iqg))**2)

                rret = minimize(guinier_objective, (0.3, 2))
                K, L = ret.x
                print("(2) K, L =", K, L)
                h_bq = - h_aq * K * phi(qv, L*2*R)

            q13 = (1.3/Rg)**2

            in_folder = get_in_folder()
            with plt.Dp():
                fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
                fig.suptitle("Hard Sphere Model Fitting Demo using %s descending side, $R_g$=%.3g" % (in_folder, Rg), fontsize=20)

                ax0 = axes[0,0]
                ax1 = axes[0,1]
                ax2 = axes[0,2]
                ax3 = axes[0,3]

                ax4 = axes[1,0]
                ax5 = axes[1,1]
                ax6 = axes[1,2]
                ax7 = axes[1,3]

                for ax in [ax0, ax2, ax4, ax6]:
                    ax.set_yscale("log")

                ax0.set_title("Apparent Log View", fontsize=16)
                ax1.set_title("Apparent Guinier View", fontsize=16)
                ax2.set_title("Measured Only View", fontsize=16)
                ax3.set_title("Linear A(q) View", fontsize=16)
                ax4.set_title("Eoii-removed Log View", fontsize=16)
                ax5.set_title("Eoii-removed Guinier View", fontsize=16)
                ax6.set_title("Model Only View", fontsize=16)
                ax7.set_title("Linear B(q) View", fontsize=16)

                ax0.plot(qv, Iq, label="Data I(q)")
                ax0.plot(qv, Hq, label="Sphere I(q)")
                ax0.legend()

                ax1.plot(qvg2, np.log(Iqg), label="Data I(q)")
                ax1.plot(qvg2, np.log(Hqg), label="Sphere I(q)")

                ax2.plot(qv, Iq, label="Data I(q)")
                ax2.plot(qv, d_aq, label="Data A(q)")
                ax2.legend()

                ax3.plot(qv, d_aq, label="Data A(q)")
                ax3.plot(qv, h_aq, label="Sphere A(q)")
                ax3.legend()

                ax4.plot(qv, d_aq, label="Data A(q)")
                ax4.plot(qv, h_aq, label="Sphere A(q)")
                ax4.legend()

                ax5.plot(qvg2, np.log(d_aqg), label="Data A(q)")
                ax5.plot(qvg2, np.log(h_aqg), label="Sphere A(q)")

                def plot_q13(ax):
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(ymin, ymax)
                    ax.plot([q13, q13], [ymin, ymax], color="yellow", label="$qR_g=1.3$")

                for ax in ax1, ax5:
                    plot_q13(ax)
                    ax.legend()

                ax6.plot(qv, Hq, label="Sphere I(q)")
                ax6.plot(qv, h_aq, label="Sphere A(q)")
                ax6.legend()

                ax7.plot(qv, bq, label="Data B(q)")
                h_bq_scaled = h_bq*scale**2
                ax7.plot(qv, h_bq_scaled, label="Sphere B(q)")

                zeros = np.zeros(len(qv))

                def bound_objective(p):
                    L = p[0]
                    return np.sum(np.max([zeros, np.abs(h_bq_scaled) - 1/(qv*L*R)**2], axis=0)**2) - L

                retb = minimize(bound_objective, (1,))
                L = retb.x[0]
                print("L=", L)

                by = 1/(qv*L*R)**2
                ax7.plot(qv, by, ":", color="red", label="B(q) bound")
                ax7.plot(qv, -by,":", color="red")

                ymin, ymax = ax7.get_ylim()
                ax7.set_ylim(-10, 10)
                ax7.legend()
                fig.tight_layout()
                plt.show()

        j2 =  np.argmax(smooth(bq_))
        slice2_ = slice(j2,i2)
        q_ = qv[slice2_]
        aq2_ = aq[slice2_]
        bq2_ = bq[slice2_]

        hK, hL = ret.x
        h_bq = -hK*phi(q_, 2*hL*R)*aq2_/c1
        upper_bq2 = np.max([bq2_, h_bq], axis=0)

        def objective2(p):
            L, = p
            bound = 1/(q_*L*R)**2 * aq2_/c1
            diff = upper_bq2 - bound
            return 99*np.mean(diff[diff>0]**2) + np.mean(diff[diff<=0]**2)

        ret2 = minimize(objective2, (hL,))
        retL = ret2.x[0]

        if debug_plot:
            K, L = ret.x
            print("L=", L, "K=", K)
            print("ret.nit=", ret.nit)
            print("ret2.x[0]=", ret2.x[0])
            h_bq = -K*phi(qv, 2*L*R)*aq/c1
            bound = 1/(qv*retL*R)**2 * aq/c1
            c2 = c1**2
            safe_aq = np.max([aq, np.ones(len(aq))*1e-6], axis=0)
            sq = 1 + bq*c1/safe_aq
            h_sq = 1 - K*phi(qv, 2*L*R)
            with plt.Dp():
                fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(18,6))
                fig.suptitle("estimate_RgL result: $R_g=%.3g$" % Rg, fontsize=20)
                ax1.set_title("Debug Plot on the B(q) basis", fontsize=16)
                ax2.set_title("Debug Plot on the S(q) basis", fontsize=16)

                ax1.plot(qv, bq*c2, label="Data B(q)")
                ax1.plot(qv, h_bq*c2, label="Sphere B(q)")
                ymin, ymax = ax1.get_ylim()
                ax1.set_ylim(ymin, ymax)
                ax1.plot(qv, -bound*c2, ":", color="red")
                ax1.plot(qv, +bound*c2, ":", color="red")
                ax1.set_xlim(0, qv[i2])
                # ax1.set_ylim(-20*c2, 20*c2)
                ax1.legend()

                ax2.plot(qv, sq, label="Data Sq(q)")
                ax2.plot(qv, h_sq, label="Sphere Sq(q)")
                ax2.set_xlim(0, qv[i2])
                ymin, ymax = ax2.get_ylim()
                ax2.set_ylim(-0.05, 5)
                ax2.legend()

                fig.tight_layout()
                plt.show()

        return Rg, R, retL, hK, hL
