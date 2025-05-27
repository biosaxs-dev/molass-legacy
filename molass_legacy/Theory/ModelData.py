"""
    Theory.ModelData.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ElutionCurveModels import emg
from .SolidSphere import a_func, b_to_a_func

class ModelData:
    def __init__(self, qv, num_el, Rg):
        self.qv = qv
        self.cx = np.arange(num_el)

        h = 1
        mu = num_el//2
        sigma = 30
        tau = 0
        self.e_params = [(h, mu, sigma, tau)]
        self.cy = emg(self.cx, h, mu, sigma, tau)
        self.Rg = Rg
        self.R = R = np.sqrt(5/3)*Rg

        self.a = a_func(qv, R)
        self.b2a = b_to_a_func(qv, R)
        self.b = self.a * self.b2a

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

def demo1():
    from MatrixData import simple_plot_3d

    fontsize = 16
    noise = 0.01

    qv = np.linspace(0.005, 0.5, 600)
    md = ModelData(qv, 300, 32)
    M, E = md.get_data(noise=noise, conc=1)

    fig = plt.figure(figsize=(21, 7))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='3d')

    fig.suptitle("Noisy Data generated with Solid Sphere Model", fontsize=20)

    ax1.set_title("Gaussian Elution Curve", fontsize=fontsize)
    ax1.plot(md.cx, md.cy, color='orange')

    ax2.set_title("Solid Sphere Scattering Curve: Rg=%g (R=%.3g)" % (md.Rg, md.R), fontsize=fontsize)
    ax2.set_yscale('log')
    ax2.plot(qv, md.a, label='A(q)')

    ax2t = ax2.twinx()
    ax2t.grid(False)
    ax2t.set_ylim(-2, 2)
    ax2t.plot(qv, md.b2a, color='yellow', label='B(q)/A(q)')
    ax2t.plot(qv, md.b, color='pink', label='B(q)')

    ax2.legend(fontsize=fontsize)
    ax2t.legend(bbox_to_anchor=(1, 0.9), loc='upper right', fontsize=fontsize)

    ax3.set_title("Generated Noisy Data (noise=%g)" % noise, fontsize=fontsize)
    simple_plot_3d(ax3, M, x=qv)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()

def demo2():
    from Rank.Synthesized import synthesized_data

    qv = np.linspace(0.005, 0.5, 100)
    qi = bisect_right(qv, 0.02)

    Rg = 32
    qv = np.linspace(0.005, 0.5, 600)
    md = ModelData(qv, 300, Rg)

    noises = [0.01, 0.03, 0.06]
    boundaries = [0.05, 0.1, 0.15, 0.2, 0.25]

    nrows = len(noises)
    ncols = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21, 11))

    fig.suptitle("Rank Boundary Evaluation with Noisy Solid Sphere Model", fontsize=20)

    for i, noise in enumerate(noises):
        for j, b in enumerate(boundaries):
            M, E = md.get_data(noise=noise, conc=1)
            c = M[qi,:]
            c = c/np.max(c)
            C = np.array([c, c**2])
            M_ = synthesized_data(qv, M, boundary=b)
            P = M_ @ np.linalg.pinv(C)
            a_ = P[:,0]
            b_ = P[:,1]
            ax = axes[i,j]
            ax.plot(qv, a_)
            axt = ax.twinx()
            axt.grid(False)
            axt.plot(qv, b_, color='pink')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    plt.show()
