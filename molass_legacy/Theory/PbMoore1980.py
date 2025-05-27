"""
    Theory.PbMoore1980.py

    Basic functions shannon_channels, Bt, Ct, Yt, ...
    are borrowed from saxstats.py in DENSS

        Author: Thomas D. Grant
        Email:  <tdgrant@buffalo.edu>
        Alt Email:  <tgrant@hwi.buffalo.edu>
        Copyright 2017, 2018, 2019, 2020 The Research Foundation for SUNY
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

"""
    Y = 4*A@C
    A = 1/4*Y@C⁻¹
"""

def shannon_channels(D, qmax=0.5, qmin=0.0):
    """Return the number of Shannon channels given a q range and maximum particle dimension"""
    width = np.pi / D
    num_channels = int((qmax-qmin) / width)
    return num_channels

def Bt(n, q, D):
    return (n*np.pi)**2/((n*np.pi)**2-(q*D)**2) * np.sinc(q*D/np.pi) * (-1)**(n+1)
    # return np.pi*n*D*(-1)**(n+1)*np.sin(2*np.pi*D*q)/((np.pi*n)**2 - (2*np.pi*D*q)**2)

def Yt(I, Ierr, Bm):
    """Return the values of Y, an m-length vector."""
    return np.einsum('i, ki->k', I/Ierr**2, Bm)

def Ct(Ierr, Bm, Bn):
    """Return the values of C, a m x n variance-covariance matrix"""
    return np.einsum('ij,kj->ik', Bm/Ierr**2, Bn)

def shannon_intensities(q, I, Ierr, D, ne=2):
    """Calculate Shannon intensities from experimental I(q) profile."""

    nq = len(q)
    qmin = q[0]
    qmax = q[-1]
    n = shannon_channels(qmax, D) + ne

    B = np.zeros((n, nq))
    C = np.zeros((n, n))
    Y = np.zeros((n))

    Ni = np.arange(n)
    N = Ni + 1
    Mi = Ni.copy()
    M = N.copy()
    qn = np.pi/D * N

    B[Ni] = Bt(N[:, None], q, D)
    Y[Mi] = Yt(I, Ierr, B[Mi])
    C[Mi[:, None], Ni] = Ct(Ierr, B[Mi], B[Ni])
    Cinv = np.linalg.inv(C)
    In = 0.5 * np.linalg.solve(C, Y)
    Inerr = 0.5 * (np.diagonal(Cinv))**(0.5)
    return qn, In, Inerr

def Ish2Iq(Ish, D, q=(np.arange(500)+1.)/1000):
    """Calculate I(q) from intensities at Shannon points."""
    q = np.atleast_1d(q)
    Ish = np.atleast_1d(Ish)
    Iq = np.zeros((len(q),2))
    Iq[:,0] = q
    n = len(Ish)
    N = np.arange(n)+1
    denominator = (N[:,None]*np.pi)**2-(q*D)**2
    I = 2*np.einsum('k,ki->i',Ish,(N[:,None]*np.pi)**2 / denominator * np.sinc(q*D/np.pi) * (-1)**(N[:,None]+1))
    Iq[:,1] = I
    return Iq

def ift_intensities(q, I, Ierr, D=None, qc=None, r=None, ne=2):
    if D is None:
        from molass_legacy.DENSS.saxstats.saxstats import estimate_dmax
        Iq = np.array([q, I, Ierr]).T
        D = estimate_dmax(Iq)[0]
        print("estimated dmax = %.4g" % D)

    qn, In, Inerr = shannon_intensities(q, I, Ierr, D, ne=ne)

    if qc is None:
        qc = q
    Ic = Ish2Iq(In, D, q=qc)[:,1]
    Icerr = np.interp(qc, q, Ierr)
    return qc, Ic, Icerr, D

"""
    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
def demo(root, sd, pno=0, debug=True, crysol_int_files=None):
    from bisect import bisect_right
    from .SynthesizedLRF import synthesized_lrf_spike as synthesized_lrf
    from SvdDenoise import get_denoised_data
    from .Rg import compute_corrected_Rg, compute_Rg
    from .SolidSphere import get_boundary_params_simple

    assert crysol_int_files is not None

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

    pt = M[:,p]
    spt = P[:,0]

    qc2, Ic2, Icerr2, D = ift_intensities(qv, spt, E[:,p])

    i = bisect_right(qv, b1)
    pt_ = pt*Ic2[i]/pt[i]
    ptc = np.hstack([Ic2[0:i], pt_[i:]])
    qc1, Ic1, Icerr1, _ = ift_intensities(qv, ptc, E[:,p], D=D)

    ncols = len(crysol_int_files) + 1

    fig, axes = plt.subplots(ncols=ncols, figsize=(7*ncols,7))
    fig.suptitle("Difference between ALMERGE and SynLRF with IFT from P. B. Moore 1980", fontsize=20)

    ax1, ax2 = axes[0:2]
    ax1.set_title("Peak Top, LRF and PBMoore", fontsize=16)
    if len(axes) == 2:
        ax2.set_title("PBMoore and CRYSOL", fontsize=16)
        ax3 = None
    else:
        ax3 = axes[2]
        ax2.set_title("PBMoore and CRYSOL2", fontsize=16)
        ax3.set_title("PBMoore and CRYSOL3", fontsize=16)

    for ax in axes:
        ax.set_yscale('log')

    ax1.plot(qv, pt_, label='Peak Top Data')
    ax1.plot(qv, spt, label='Syn LRF')

    ymin, ymax = ax1.get_ylim()

    for ax in axes:
        ax.set_ylim(ymin, ymax)
        ax.plot(qc1, Ic1, label="ALMERGE like IFT", color='C2')
        ax.plot(qc2, Ic2, label="Syn LRF IFT", color='C3')

    if crysol_int_files is not None:
        from CrysolUtils import np_loadtxt_crysol
        for j, ax in enumerate(axes[1:]):
            crysol_int = crysol_int_files[j]
            name = crysol_int[0:4]
            data, _ = np_loadtxt_crysol(crysol_int)
            print('data.shape=', data.shape)
            if data.shape[1] == 5:
                crysol_ver = 2
                K = 2
            else:
                crysol_ver = 3
                K = 2
            qv = data[:,0]
            for k in range(1,K):
                y = data[:,k].copy()
                y *= Ic2[0]/y[0]
                if ax3 is None:
                    ax.plot(qv, y, label='%s-CRYSOL' % (name), color='C%d' % (3+k))
                else:
                    # older style when there were tow versions of crysol, i.e., crysol and crysol_30
                    ax.plot(qv, y, label='%s-CRYSOL%d-%d' % (name, crysol_ver, k), color='C%d' % (3+k))

    for ax in axes:
        ax.plot([b1, b1], [ymin, ymax], ':', color='gray', label='Rank boundary')
        ax.legend()

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()
