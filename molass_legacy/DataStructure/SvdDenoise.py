"""
    SvdDenoise.py

    Copyright (c) 2018-2020, 2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass.LowRank.LowRankInfo import get_denoised_data     # for forward compatibility

def get_denoised_error(M_, M, E, left=True, debug=False, q=None, eno=None ):

    """
        M_ = T @ M
        T = M_ @ Mpinv
    """

    Mpinv = np.linalg.pinv( M )
    if left:
        W = np.dot( M_, Mpinv )
        E_ = np.sqrt( np.dot( W**2, E**2) )
    else:
        W = np.dot( Mpinv, M )
        E_ = np.sqrt( np.dot( E**2, W**2) )

    if debug and eno is not None:
        debug_plot(q, M, E, M_, E_, eno=eno)

    return E_

def debug_plot(x, M, E, M_, E_, eno):
    import molass_legacy.KekLib.DebugPlot as plt
    fig = plt.figure(figsize=(21,7))
    ax0  = fig.add_subplot(131)
    ax1  = fig.add_subplot(132)
    ax2  = fig.add_subplot(133)

    ax0.set_title("Error propagation in denoising with SVD using MR-inverve")
    ax0.axes.get_xaxis().set_ticks([])
    ax0.axes.get_yaxis().set_ticks([])
    ax0.text(0.1,0.85, r"$M = U \cdot \Sigma \cdot V^T$", fontsize=30)
    ax0.text(0.1,0.65, r"$\widetilde{M} = \widetilde{U} \cdot \widetilde{\Sigma} \cdot  \widetilde{V}^T $", fontsize=30)
    ax0.text(0.1,0.45, r"$\widetilde{M} \approx  W \cdot M$", fontsize=30)
    ax0.text(0.1,0.25, r"$W \approx  W \cdot M \cdot M^+ \approx  \widetilde{M} \cdot M^+$", fontsize=30)
    ax0.text(0.1,0.05, r"$\widetilde{E} = \sqrt{ W^{(2)} \cdot E^{(2)} }$", fontsize=30)
    """
        [ a b ]  [x]
        [ c d ]  [y]
        ax + by
        cx + dy
    """

    ax1.set_title( "Propagated errors in linear scale at eno=%d" % eno )
    ax1.set_xlabel('Q')
    ax1.set_ylabel('Intensity')
    ax1.plot(x, M[eno, :], label='data')
    ax1.plot(x, E[eno, :], label='error')
    ax1.plot(x, M_[eno, :], label='denoised data')
    ax1.plot(x, E_[eno, :], label='denoised error')
    ax1.legend()
    ax2.set_title( "Propagated errors in $Log_{10}$ scale at eno=%d" % eno )
    ax2.set_xlabel('Q')
    ax2.set_ylabel('$Log_{10}(Intensity)$')
    ax2.plot(x, np.log10(M[eno, :]), label='data')
    ax2.plot(x, np.log10(E[eno, :]), label='error')
    ax2.plot(x, np.log10(M_[eno, :]), label='denoised data')
    ax2.plot(x, np.log10(E_[eno, :]), label='error of denoised data')
    ax2.legend()
    fig.tight_layout()
    plt.show()
