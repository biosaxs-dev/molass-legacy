"""
    SimData.py

    Copyright (c) 2020-2025, SAXS Team, KEK-PF
"""
from bisect import bisect_right
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from molass_legacy.DENSS.DenssUtils import fit_data_impl
from molass_legacy.Models.ElutionCurveModels import emg

class SimData:
    def __init__(self, a_file, b_file, num_elutions=300,  debug=False):
        A = np.loadtxt(a_file)
        q = A[:,0]
        a = A[:,1]
        e = A[:,2]
        sasrec, work_info = fit_data_impl(q, a, e, file=a_file, D=76)
        qc = sasrec.qc
        ac = sasrec.Ic
        ec = work_info.Icerr
        spline = UnivariateSpline(qc, ac, s=0)
        self.q = q
        self.a = spline(q)
        self.ae = e

        B = np.loadtxt(b_file)
        q = B[:,0]
        b = B[:,1]
        e = B[:,2]
        num_nkots = 10
        knots = np.linspace( 0, q[-1], num_nkots + 2 )
        spline = LSQUnivariateSpline(q, b, knots[1:-1], ext=3)
        self.b = spline(q)
        self.be = e

        self.i = bisect_right(q, 0.02)

        num_elutions = 300
        h = 1
        mu = num_elutions//2
        sigma = 30
        tau = 0
        self.cx = x = np.arange(num_elutions)
        self.e_params = [(h, mu, sigma, tau)]
        self.cy = emg(x, h, mu, sigma, tau)

        if debug:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))
            ax1.set_yscale('log')
            ax1.plot(q, a)
            ax1.plot(qc, ac)

            ax2.set_yscale('log')
            ax2.plot(q, a)
            ax2.plot(q, self.a)

            ax3.plot(q, b)
            ax3.plot(q, self.b)

            fig.tight_layout()
            plt.show()

    def get_data(self, noise=0.01, conc=0, seed=None):
        c = self.cy/np.max(self.cy)
        if conc > 0:
            P = np.array([self.a, self.b*conc]).T
            C = np.array([c, c**2])
        else:
            P = self.a[:, np.newaxis]
            C = c[np.newaxis,:]

        M = P @ C
        print('M.shape', M.shape)
        M_ = M/np.max(M)

        if seed is not None:
            np.random.seed(seed)

        if noise > 0:
            GE = np.random.normal(0, 1, M.shape)*noise/M_
            M *= (1 + GE)
            E = np.abs(M*GE)

        return M, E

    def demo_plot(self, axes=None, show=True):
        from MatrixData import simple_plot_3d
        from SvdDenoise import get_denoised_data
        from molass_legacy.Models.ElutionCurveModels import emg_x_from_height_ratio
        from matplotlib.patches import Rectangle

        noise = 0.01
        cd = 1
        M, E = self.get_data(noise=noise, conc=cd)
        c = self.cy/np.max(self.cy)
        C = np.array([c, c**2])

        h, mu, sigma, tau = self.e_params[0]
        f, t = [int(p+0.5) for p in emg_x_from_height_ratio(0.5, mu, sigma, tau)]
        print("(f, t)=", (f, t))
        eslice = slice(f,t+1)
        M_ = get_denoised_data(M[:,eslice], rank=2)
        C_ = C[:,eslice]
        P_ = M_ @ np.linalg.pinv(C_)

        if axes is None:
            fig = plt.figure(figsize=(16, 11))
            ax0 = fig.add_subplot(221)
            ax1 = fig.add_subplot(222)
            ax2 = fig.add_subplot(223, projection='3d')
            ax3 = fig.add_subplot(224)

        else:
            ax1, ax2, ax3 = axes

        ax0.set_title("Gaussian Elution Curve", fontsize=16)
        ax0.plot(self.cx, self.cy, color='orange')

        ax1.set_title("Smoothed Extrapolated Curves from 20180526/OA", fontsize=16)
        ax1.set_yscale('log')
        ax1.plot(self.q, self.a, color='C1', label='A(q)')

        ax1t = ax1.twinx()
        ax1t.grid(False)
        ax1t.plot(self.q, self.b, color='pink', label='B(q)')

        ax1.legend()
        ax1t.legend(bbox_to_anchor=(1, 0.95), loc='upper right')

        ax2.set_title("Generated Data with Noise=%g, SCD=%d" % (noise, cd), fontsize=16)
        simple_plot_3d(ax2, M)

        ymin, ymax = ax0.get_ylim()
        p = Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax0.add_patch(p)

        ax3.set_title("Solved Extrapolated Curves", fontsize=16)
        ax3.set_yscale('log')
        ax3.plot(self.q, P_[:,0], color='C1', label='A(q)')

        ax3t = ax3.twinx()
        ax3t.grid(False)
        ax3t.plot(self.q, P_[:,1], color='pink', label='B(q)')

        ax3.legend()
        ax3t.legend(bbox_to_anchor=(1, 0.95), loc='upper right')

        fig.tight_layout()

        if show:
            plt.show()
