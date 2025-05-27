"""
    Theory.RdfCheck.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy.stats import linregress
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from PlotUtils import align_yaxis_np
from MatrixData import simple_plot_3d
from SvdDenoise import get_denoised_data
from .SolidSphere import phi
from .Rg import compute_corrected_Rg
from Rank.Synthesized import synthesized_data
from Theory.SolidSphere import get_boundary_params, get_boundary_params_simple
from DataUtils import get_in_folder

def demo(parent, sd, pno=0, use_estimated_c=False, use_better_estimate=True, debug=False):
    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[pno]
    f_ = range_[0]
    p_ = range_[1]
    t_ = range_[2]

    f, t = f_, t_
    eslice = slice(f,t)

    Rg, gf, gt, _ = compute_corrected_Rg(sd, ecurve, pno, qv, M[:,eslice], E[:,eslice])
    aslice = slice(0, gt)

    plt.push()
    fig = plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    fig.suptitle("Spherical Fitting Demo for " + get_in_folder(), fontsize=30)
    ax1.set_title("LRF Range", fontsize=20)
    ax2.set_title("B(q)/A(q)", fontsize=20)
    ax3.set_title("g(r) - 1", fontsize=20)

    b1, b2, k = get_boundary_params_simple(Rg)
    M_, _ = synthesized_data(qv, M[:,eslice], Rg, cd=2, boundary=b1, k=k)

    i = bisect_right(qv, 0.05)
    c = M[i,:]
    c_ = c[eslice]
    C = np.array([c_, c_**2])
    P = M_ @ np.linalg.pinv(C)

    N = len(qv)

    bq0 = int(N*0.8)
    Bq = P[:,1]
    # Bq -= np.average(Bq[bq0:])

    sq_1 = Bq/P[:,0]

    ax1.plot(c, color='orange')

    ymin, ymax = ax1.get_ylim()
    p = Rectangle(
            (f, ymin),      # (x,y)
            t - f,          # width
            ymax - ymin,    # height
            facecolor   = 'cyan',
            alpha       = 0.2,
        )
    ax1.add_patch(p)

    ax2.plot(qv, sq_1)

    Dmax = 70
    Nr = 200
    dr = Dmax/Nr
    rv = np.linspace(0, Dmax, Nr) + dr/2
    dq = qv[-1]/N
    gr_1 = np.zeros(Nr)

    for k, r in enumerate(rv):
        gr_1[k] = 1/(4*np.pi) * np.sum( sq_1 * np.sinc(qv*r) * qv**2 ) * dq 

    ax3.plot(rv, gr_1)

    sq_1_f = np.zeros(N)
    for k, q in enumerate(qv):
        sq_1_f[k] = 4*np.pi * np.sum( gr_1 * np.sinc(q*rv) * rv**2 ) * dr

    ax2t = ax2.twinx()
    ax2t.plot(qv, sq_1_f)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()
    plt.pop()
