"""
    Theory.Prism.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

USE_PRISM = False
USE_PYOZ = False

def compute_rdf_prism():
    import pyPRISM
    sys = pyPRISM.System(['polymer'],kT=1.0)
    sys.domain = pyPRISM.Domain(dr=0.05,length=4096)
    sys.diameter['polymer'] = 1.0
    sys.density['polymer'] = 0.9
    sys.closure['polymer','polymer'] = pyPRISM.closure.PercusYevick()
    sys.potential['polymer','polymer'] = pyPRISM.potential.HardSphere()
    sys.omega['polymer','polymer'] = pyPRISM.omega.GaussianRing(sigma=sys.diameter['polymer','polymer'],length=2000)
    PRISM = sys.createPRISM()
    guess = np.zeros_like(sys.domain.r)
    result = PRISM.solve(guess)
    x = sys.domain.r
    y = pyPRISM.calculate.pair_correlation(PRISM)['polymer','polymer']
    return x, y

def compute_rdf_pyoz():
    import pyoz as oz
    lj_unary = oz.System()
    lj_unary.set_interaction(0, 0, oz.lennard_jones(lj_unary.r, eps=1.0, sig=1.0))
    r = lj_unary.r
    U_r = lj_unary.U_r
    g_r, c_r, e_r, h_k = lj_unary.solve(rhos=0.1, closure_name='hnc')
    return r, g_r[0,0]

def compute_sq(qv, r, g, rho=0.01):
    sy = np.zeros(len(qv))
    r2 = r**2
    scale = 4 * np.pi * rho
    for k, q in enumerate(qv):
        sy[k] = 1 + scale * np.sum((g - 1)*np.sinc(q*r)*r2)
    return sy

def compute_rdf(qv, sy, rv, rho=0.01):
    gy = np.zeros(len(rv))
    scale = 1/(2*np.pi**2*rho)
    q2 = qv**2
    for i, r in enumerate(rv):
        gy[i] = 1 + scale * np.sum((sy - 1)*np.sinc(qv*r)*q2)
    return gy

def demo(sp, sd, pno=0):
    from bisect import bisect_right
    from Theory.Rg import compute_corrected_Rg
    from SvdDenoise import get_denoised_data
    from Theory.SolidSphere import phi, get_boundary_params_simple
    from Theory.SynthesizedLRF import synthesized_lrf_spike
    from .DjKinning1984 import P0, S0

    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[pno]
    f = range_[0]
    p = range_[1]
    t = range_[2]

    pt = M[:,p]

    x = ecurve.x
    y = ecurve.y

    eslice = slice(f,t)
    M0 = M[:,eslice]
    Rg, gf, gt, _ = compute_corrected_Rg(sd, ecurve, pno, qv, M0, E[:,eslice])
    R = np.sqrt(5/3)*Rg

    if True:
        i = np.argmax(pt)
    else:
        # q*5.4*R/2 == np.pi/2
        q0 = np.pi/(5.4*R)
        print("q0=", q0)
        i = bisect_right(qv, q0)

    p0 = P0(qv, R)
    s0 = S0(qv, R/2*5.4, K=1)

    norm_list = []

    c_ = y[eslice]

    M2 = get_denoised_data(M0, rank=2)
    C2 = np.array([c_, c_**2])
    P2 = M2 @ np.linalg.pinv(C2)

    b1, b2, k = get_boundary_params_simple(Rg)
    P_ = synthesized_lrf_spike(qv, M0, c_, M2, P2, boundary=b1, k=k)
    y1 = P_[:,0]
    y1n = y1/y1[0]
    s1 = 1 + P_[:,1]/P_[:,0]

    rho = 0.005

    if USE_PRISM:
        gxp, gyp = compute_rdf_prism()
        s2r = compute_sq(qv, gxp, gyp)

    gx = np.linspace(0, 500, 1000)
    g0 = compute_rdf(qv, s0, gx, rho=rho)
    g1 = compute_rdf(qv, s1, gx, rho=rho)

    if USE_PYOZ:
        gx_, g_ = compute_rdf_pyoz()
        s_r = compute_sq(qv, gx_, g_, rho=rho)

    s0r = compute_sq(qv, gx, g0, rho=rho)
    s1r = compute_sq(qv, gx, g1, rho=rho)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21,11))
    ax1, ax2, ax3 = axes[0,:]
    ax4, ax5, ax6 = axes[1,:]

    fig.suptitle("S(q) recomputation from g(r)", fontsize=20)
    ax1.set_title("S(q)", fontsize=16)
    ax2.set_title("P(q)*S(q)", fontsize=16)
    ax3.set_title("g(r)", fontsize=16)
    ax4.set_title("S(q) recomputed from g(r)", fontsize=16)

    ax2.set_yscale('log')

    label0 = "Hard Sphere"
    label1 = "GI (Synthesized LRF)"

    ax1.plot(qv, s0, label=label0)
    ax1.plot(qv, s1, label=label1)

    y0 = p0*s0
    ax2.plot(qv, y0, label=label0)
    y1n = y1/y1[0]
    ax2.plot(qv, y1n, label=label1)

    ax3.plot(gx, g0, label=label0)
    ax3.plot(gx, g1, label=label1)

    ax4.plot(qv, s0r, label=label0)
    ax4.plot(qv, s1r, label=label1)

    if USE_PRISM:
        ax3.plot(gxp, gyp)
        ax4.plot(qv, s2r)

    if USE_PYOZ:
        ax3.plot(gx_, g_)
        ax4.plot(qv, s_r)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
