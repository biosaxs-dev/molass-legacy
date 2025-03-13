"""
    Models.Stochastic.MobileDispersion.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize

USE_SPLINE = True       # using spline while the curve model is not sufficient
M1_WEIGHT = 6
M2_WEIGHT = 4

class DispersionRatio:
    def __init__(self, rgmin, rgmax, N, T, poresize, t0, N0):
        self.poresize = poresize
        me = 1.5
        mp = 1.5
        rgs_ = np.linspace(rgmin, rgmax, 20)
        rhov = rgs_/poresize
        rhov[rhov > 1] = 1
        niv = N*(1 - rhov)**me
        tiv = T*(1 - rhov)**mp
        dsv = 2*niv*tiv**2
        dmv = (t0 + niv*tiv)**2/N0
        ratios = np.sqrt(dmv/dsv)
        if USE_SPLINE:
            self.spline = UnivariateSpline(rgs_, ratios, s=0)
        else:
            def objective(p):
                a, b = p
                y = a*np.power(self.poresize - rgs_, b)     # this curve model does not seem to be sufficient.
                return np.sum((y - ratios)**2)
            res = minimize(objective, [100,-2])
            self.params = res.x
            print("self.params=", self.params)
        self.rgs = rgs_
        self.ratios = ratios
        self.t0 = t0
        self.N0 = N0

    def __call__(self, rgs):
        if USE_SPLINE:
            return self.spline(rgs)
        else:
            a, b = self.params
            return a*np.power(self.poresize - rgs, b)

    def estimate_N0(self, mdscale):
        return self.N0/mdscale**2

    def estimete_tI(self, peak_rgs, props, target_moments_list, mdscale, N, K, x0, poresize, N0):
        me = 1.5
        mp = 1.5
        T = K/N
        rhov = peak_rgs/poresize
        rhov[rhov > 1] = 1

        def objective(p):
            tI = p[0]
            dev_list = []
            for k, (M, rho) in enumerate(zip(target_moments_list, rhov)):
                M1_ = x0 + N * T * (1 - rho)**(me + mp)
                M2_ = np.sqrt(2 * N * T**2 * (1 - rho)**(me + 2*mp) + (M1_ - tI)**2/N0)
            dev_list.append(M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_- M[1])**2)
            return np.sum(np.asarray(dev_list)*props)
        
        init_tI = x0 - self.t0
        res = minimize(objective, [init_tI])
        tI = res.x[0]
        print("init_tI, tI=",init_tI, tI)
        return tI

    def covert_to_curveparams(self, peak_rgs, props, target_moments_list, colparams, scales, debug_info=None):
        from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCOL_PARAMS
        print("covert_to_curveparams: colparams=", colparams)
        N, K, x0, poresize, mdscale = colparams[0:NUM_SDMCOL_PARAMS]
        N0 = self.estimate_N0(mdscale)
        tI = self.estimete_tI(peak_rgs, props, target_moments_list, mdscale, N, K, x0, poresize, N0)
        curveparams = np.concatenate([[N, K, x0, poresize, N0, tI], scales])
        if debug_info is not None:
            from molass_legacy.Models.Stochastic.DispersiveUtils import compute_elution_curves
            import molass_legacy.KekLib.DebugPlot as dplt
            print("mdscale=", mdscale)
            x, y = debug_info
            cy_list, ty = compute_elution_curves(x, curveparams, peak_rgs)
            with dplt.Dp():
                fig, ax = dplt.subplots()
                ax.set_title("covert_to_curveparams")
                ax.plot(x, y, label='data')
                for k, cy in enumerate(cy_list):
                    ax.plot(x, cy, ':', label='rg=%g' % peak_rgs[k])
                ax.plot(x, ty, ':', label='model total', color='red')
                ax.legend()
                fig.tight_layout()
                ret = dplt.show()
                if not ret:
                    return
        return curveparams

def proof_demo():
    from molass_legacy.Models.Stochastic.DispersivePdf import N0, dispersive_monopore_pdf

    N = 2000
    T = 0.5
    K = N*T
    me = 1.5
    mp = 1.5
    poresize = 80
    rgs = np.flip(np.arange(10, poresize, 10))
    rhov = rgs/poresize
    rhov[rhov > 1] = 1
    t0 = 500
    tI = 0
    x = np.arange(0, 1400)

    dratio = DispersionRatio(rgs[-1], rgs[0], N, T, poresize, t0, N0)

    rgv = np.linspace(rgs[-1], rgs[0], 100)

    def compute_curves(x, N, T, x0, tI, N0, rhov):
        cy_list = []
        nt_list = []
        for rho in rhov:
            ni = N*(1 - rho)**me
            ti = T*(1 - rho)**mp
            nt_list.append((ni, ti))
            cy = dispersive_monopore_pdf(x - tI, ni, ti, N0, x0 - tI)
            cy_list.append(cy)
        return cy_list, nt_list

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,8))
    fig.suptitle("Analysis of Dispersions with Dispersion Ratio Spline in Stochastic Dispersive Model", fontsize=20)

    def plot_curves_and_dispersions(ax1, ax2, ax3, N0, plot_spline=True, ref_ax=None):
        ax1.set_title("Elution Curves when Poresize=%g, $N_0$=%d" % (poresize, N0), fontsize=14)
        ax1.axvline(x=t0, label="$t_0$", color='red', alpha=0.5)
        cy_list, nt_list = compute_curves(x, N, T, t0, tI, N0, rhov)
        for k, (cy, rg) in enumerate(zip(cy_list, rgs)):
            ax1.plot(x, cy, label='$R_g$=%g' % rg, alpha=0.5)
        ax1.legend(loc='upper left')
        if ref_ax is not None:
            ax1.set_ylim(ref_ax.get_ylim())

        ax2.set_title("Mobile / Stationary Dispersions vs. $R_g$'s", fontsize=14)
        ax2.set_xlabel("$R_g$")
        ax2.set_ylabel("Dispersion")
        ax2.invert_xaxis()

        dispersions = []
        for k, (ni, ti) in enumerate(nt_list):
            ds = 2*ni*ti**2
            dm = (t0 + ni*ti)**2/N0
            dispersions.append((ds, dm))
        dispersions = np.array(dispersions)
        sdisps = np.sqrt(dispersions[:,0])
        mdisps = np.sqrt(dispersions[:,1])
        ax2.plot(rgs, sdisps, "o", label="Stationary", alpha=0.5)
        ax2.plot(rgs, mdisps, "o", label="Mobile", alpha=0.5)
        real_ratios = mdisps/sdisps
        model_ratios = dratio(rgs)
        scale = np.average(real_ratios/model_ratios)
        ax2.plot(rgs, sdisps*dratio(rgs)*scale, "+", label="Mobile (estimated with the spline)", markersize=10)
        ax2.legend()

        if plot_spline:
            ax3.set_title("Mobile / Stationary Dispersion Ratio Spline", fontsize=14)
            ax3.set_xlabel("$R_g$")
            ax3.set_ylabel("Dispersion Ratio")
            ax3.invert_xaxis()

            ax3.plot(rgs, mdisps/sdisps, "o", color="C2")
            ax3.plot(rgv, dratio(rgv), ":", color="C2")
        else:
            ax3.set_axis_off()

    plot_curves_and_dispersions(*axes[0,:], N0)
    plot_curves_and_dispersions(*axes[1,:], N0//4, plot_spline=False, ref_ax=axes[0,0])

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    sys.path.append("../lib")
    import os
    import seaborn as sns
    sns.set_theme()

    proof_demo()