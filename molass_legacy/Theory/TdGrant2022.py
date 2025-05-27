"""
    Theory.TdGrant2022.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.DENSS.DenssUtils import fit_data
from DataUtils import get_in_folder

def demo(root, sd, pno=None, debug=True, crysol_int_files=None, use_bounded_lrf=True, use_almerge=False, rank=2):
    from bisect import bisect_right
    from .SynthesizedLRF import synthesized_lrf_spike as synthesized_lrf
    from SvdDenoise import get_denoised_data
    from .Rg import compute_corrected_Rg, compute_Rg
    from .SolidSphere import get_boundary_params_simple

    assert crysol_int_files is not None

    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    if pno is None:
        pno = ecurve.primary_peak_no

    range_ = ecurve.get_ranges_by_ratio(0.5)[pno]
    f = range_[0]
    p = range_[1]
    t = range_[2]

    eslice = slice(f,t)
    x = ecurve.x
    y = ecurve.y

    M0 = M[:,eslice]

    Rg, gf, gt, _ = compute_corrected_Rg(sd, ecurve, pno, qv, M0, E[:,eslice])
    b1, b2, k = get_boundary_params_simple(Rg)
    print("Rg=%g, b1=%g" % (Rg, b1))

    c_ = y[eslice]

    if rank == 2:
        M2 = get_denoised_data(M0, rank=2)
        C2 = np.array([c_, c_**2])
        P2 = M2 @ np.linalg.pinv(C2)

        if use_bounded_lrf:
            from BoundedLRF.BoundedLrfSolver import BoundedLrfSolver
            solver = BoundedLrfSolver(qv, M2, E[:,eslice], C=C2, i=sd.xray_index)
            lrf_result = solver.solve()
            # P_, C__, Rg, R_, L_, hK, hL, bq_bounds_, coerced_bq_ = lrf_result
            P = lrf_result[0]
        else:
            P = synthesized_lrf(qv, M0, c_, M2, P2, boundary=b1, k=k)

    elif rank == 1:
        M1 = get_denoised_data(M0, rank=1)
        C1 = np.array([c_])
        P1 = M1 @ np.linalg.pinv(C1)
        P = P1
    else:
        assert False

    pt = M[:,p]
    spt = P[:,0]

    qc2, Ic2, Icerr2, D = fit_data(qv, spt, E[:,p])

    if use_almerge:
        pass
    else:
        def compute_almerge_like_values(b):
            i = bisect_right(qv, b)
            j = bisect_right(qc2, b)
            pt_ = pt*Ic2[j]/pt[i]
            ptc = np.hstack([Ic2[j-i:j], pt_[i:]])
            qc1, Ic1, Icerr1, _ = fit_data(qv, ptc, E[:,p], D=D)
            return pt_, qc1, Ic1

        pt_, qc1, Ic1 = compute_almerge_like_values(b1)

    ncols = len(crysol_int_files) + 2

    fig, axes = plt.subplots(ncols=ncols, figsize=(7*ncols,7))
    lrf_text = "Bounded LRF" if use_bounded_lrf else "SynLRF"
    in_folder = get_in_folder()
    crysol_int = crysol_int_files[0]
    crysol_int_name = crysol_int.split('\\')[-1]
    fig.suptitle("%s vs. ALMERGE on %s, %s (using IFT in DENSS)" % (lrf_text, in_folder, crysol_int_name), fontsize=20)

    ax0, ax1, ax2 = axes[0:3]

    ax0.set_title("Used Elution Range", fontsize=16)
    ax0.plot(x, y)
    ymin, ymax = ax0.get_ylim()
    p = Rectangle(
            (f, ymin),      # (x,y)
            t - f,          # width
            ymax - ymin,    # height
            facecolor   = 'cyan',
            alpha       = 0.2,
        )
    ax0.add_patch(p)
    ax0.set_xlabel("Eno")
    ax0.set_ylabel("Intensity")

    ax1.set_title("Peak Top, LRF and TdGrant", fontsize=16)
    if len(axes) == 3:
        ax2.set_title("TdGrant and CRYSOL", fontsize=16)
        ax3 = None
    else:
        ax3 = axes[3]
        ax2.set_title("TdGrant and CRYSOL2", fontsize=16)
        ax3.set_title("TdGrant and CRYSOL3", fontsize=16)

    axes_ = axes[1:]

    for ax in axes_:
        ax.set_yscale('log')

    ax1.plot(qv, pt_, label='Peak Top Data')
    label = 'Bounded LRF' if use_bounded_lrf else 'Syn LRF'
    ax1.plot(qv, spt, label=label)

    ymin, ymax = ax1.get_ylim()

    for ax in axes_:
        ax.set_ylim(ymin, ymax)
        if Ic1 is not None:
            ax.plot(qc1, Ic1, label="ALMERGE like IFT", color='C2')
        label = 'Bounded LRF IFT' if use_bounded_lrf else 'Syn LRF IFT'
        ax.plot(qc2, Ic2, label=label, color='C3')

    if crysol_int_files is not None:
        from CrysolUtils import np_loadtxt_crysol
        for j, ax in enumerate(axes_[1:]):
            # crysol_int = crysol_int_files[j]
            # name = crysol_int.split('\\')[-1][0:4]
            name = crysol_int_name[0:4]
            data, _ = np_loadtxt_crysol(crysol_int)
            print('data.shape=', data.shape)
            if data.shape[1] == 5:
                crysol_ver = 2
                K = 2
            else:
                crysol_ver = 3
                K = 2
            qv_ = data[:,0]
            for k in range(1,K):
                y = data[:,k].copy()
                y *= Ic2[0]/y[0]
                if ax3 is None:
                    ax.plot(qv_, y, label='%s-CRYSOL' % (name), color='C%d' % (3+k))
                else:
                    # older style when there were tow versions of crysol, i.e., crysol and crysol_30
                    ax.plot(qv_, y, label='%s-CRYSOL%d-%d' % (name, crysol_ver, k), color='C%d' % (3+k))

    for ax in axes_:
        if rank == 2:
            ax.plot([b1, b1], [ymin, ymax], ':', color='gray', label='Rank boundary')
        ax.set_xlabel("Q")
        ax.set_ylabel("Intensity")
        ax.legend()

    if rank == 2 and use_bounded_lrf:
        for ax in axes_:
            axt = ax.twinx()
            axt.grid(False)
            for k, bound in enumerate(lrf_result[7]):   # bq_bounds_
                label = None if k > 0 else "B(q) bounds"
                axt.plot(qv, bound, ":", color="red", label=label)
            axt.legend(loc="lower left")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()
