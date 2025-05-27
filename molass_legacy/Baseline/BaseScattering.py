# coding: utf-8
"""
    Baseline.BaseScattering.py

    Copyright (c) 2020-2025, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.gridspec import GridSpec
from molass_legacy.DataStructure.MatrixData import simple_plot_3d
from .Baseline import compute_baseline

TITLE_FONTSIZE = 16

def primary_peak_topx(ecurve):
    return ecurve.primary_peak_i

def old_demo(root, in_folder, show_preview=False):
    import molass_legacy.KekLib.DebugPlot as plt
    from MeasuredData import MeasuredData

    print(in_folder)
    md = MeasuredData(in_folder)

    if show_preview:
        from molass_legacy.Tools.ThreeDimViewer import ThreeDimViewer
        dialog = ThreeDimViewer(root, md)
        dialog.show()

    xr = md.xr
    xr.set_elution_curve()
    i_slice = xr.i_slice
    j_slice = xr.j_slice
    i_index = xr.e_index - i_slice.start

    whole_data = xr.data
    data = whole_data[i_slice,j_slice]
    q = xr.vector[i_slice]

    fig = plt.figure(figsize=(21, 11))
    gs = GridSpec(2,3)
    ax00 = fig.add_subplot(gs[0,0], projection='3d')
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax10 = fig.add_subplot(gs[1,0])
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])

    ax00.set_title("3D Data View", y=1.09, fontsize=TITLE_FONTSIZE)
    simple_plot_3d(ax00, data, x=q)
    z = xr.e_curve.y[j_slice]
    start = 0 if j_slice.start is None else j_slice.start
    topx = primary_peak_topx(xr.e_curve) - start
    print('topx=', topx)

    q1 = xr.vector[i_index]
    size = data.shape[1]
    x = np.ones(size)*q1
    y = np.arange(size)

    yp = np.average(data[:,topx-10:topx+10], axis=1)
    yd = np.average(data[:,0:50], axis=1) - np.average(data[:,-50:], axis=1)
    scale = yp[i_index]/yd[i_index]
    yd_ = yd*scale

    ax00.plot(x, y, z, color='orange')

    baseline = None
    for ax1, ax2, integral in [
            (ax01, ax02, False), 
            (ax11, ax12, True),
            ]:
        if integral:
            title = "LPM Integral Baseline"
        else:
            title = "LPM Linear Baseline"
        b = compute_baseline(z, integral=integral)
        if baseline is None:
            baseline = b
        z_ = z - b

        ax1.set_title(title, fontsize=TITLE_FONTSIZE)
        ax1.plot(z, color='orange')
        ax1.plot(b, ':', color='red')
        ax1.plot(topx, z[topx], 'o', color='yellow')

        ax2.plot(q, np.log10(yp), label='peak top')
        ax2.plot(q, np.log10(yd_), label='base diff')

        C = np.array([z_, b])
        P = np.dot(data, np.linalg.pinv(C))

        y_ = P[:,0]
        scale = yp[i_index]/y_[i_index]
        ax2.plot(q, np.log10(y_*scale), label='MPI peak')

        yb = P[:,1]
        scale = yp[i_index]/yb[i_index]
        ax2.plot(q, np.log10(yb*scale), label='MPI base')

        ax2.legend()

    fig.tight_layout()
    plt.show()

def demo(parent, in_folder, force_integral=False):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Mapping.MapperConstructor import create_mapper

    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_sd(debug=False)
    pre_recog = PreliminaryRecognition(sd)
    sd_ = pre_recog.get_analysis_copy()

    if force_integral:
        from molass_legacy._MOLASS.SerialSettings import set_setting
        set_setting("baseline_manually", 1)
        set_setting("xray_baseline_type", 2)

    mapper = create_mapper(parent, sd_)
    mapped_info = mapper.get_mapped_info()
    end_slices = mapper.x_curve.get_end_slices()
    basescattering_correct(mapped_info, sd_.intensity_array, end_slices=end_slices, debug=True)

def get_baseplane(baseline, end_slices, yb):
    base_hights = [np.average(baseline[s]) for s in end_slices]
    n_baseline = base_hights[0] + (baseline - base_hights[0])/(base_hights[1] - base_hights[0])
    return np.dot(yb[:,np.newaxis], n_baseline[np.newaxis,:])

def basescattering_correct(mapped_info, intensity_array, use_mpi=False, end_slices=None, debug=False):
    x_curve = mapped_info.x_curve
    x_base = mapped_info.x_base

    x = x_curve.x
    y = x_curve.y       # note that y has been already corrected.

    if debug:
        from bisect import bisect_right
        from molass_legacy._MOLASS.SerialSettings import get_xray_picking
        print("end_slices=", end_slices)
        qv = intensity_array[0,:,0]
        picking = get_xray_picking()
        i = bisect_right(qv, picking)
        with plt.Dp():
            fig, ax = plt.subplots()
            init_ey = intensity_array[:,i,1].copy()
            ax.set_title("basescattering_correct (1)")
            ax.plot(x, init_ey, label="init_ey")
            ax.plot(x, y, color='orange', label="x_curve.y")
            for s in end_slices:
                ax.plot(x[s], y[s], 'o', color='yellow')
            ax.plot(x, x_base, ':', color='red', label="mapped_info.x_base")
            ax.legend()
            fig.tight_layout()
            plt.show()

    M = intensity_array[:,:,1].T
    if use_mpi or debug:
        C = np.array([y, x_base])/x_curve.max_y
        P = np.dot(M, np.linalg.pinv(C))
        yb = P[:,1]
        # TODO: scale
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("basescattering_correct (2)")
                ax.set_yscale("log")
                for k, yp in enumerate(P.T):
                    ax.plot(qv, yp, label="P[:,%d]" % k)
                ax.legend()
                fig.tight_layout()
                plt.show()

    if not use_mpi:
        assert end_slices is not None
        yb_ = np.average(M[:,end_slices[1]], axis=1) - np.average(M[:,end_slices[0]], axis=1)
        if debug:
            print("M.shape=", M.shape, "end_slices=", end_slices)
            pti = x_curve.primary_peak_i
            qv = intensity_array[0,:,0]
            scale1 = y[pti]
            scale2 = x_base[pti]
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("basescattering_correct (3)")
                ax.set_yscale('log')
                yp = M[:,pti]
                ax.plot(qv, yp, label='peak top')
                ax.plot(qv, P[:,0]*scale1, label='MPI corrected')
                ax.plot(qv, yp - yb_, label='Diff corrected')
                ax.plot(qv, yb*scale2, label="MPI base", alpha=0.3)
                ax.plot(qv, yb_, label='Diff base', alpha=0.3)
                ax.legend()
                fig.tight_layout()
                plt.show()
        yb = yb_

    B = get_baseplane(x_base, end_slices, yb)
    intensity_array[:,:,1] -= B.T

    if debug:
        with plt.Dp():
            fig = plt.figure(figsize=(14,7))
            fig.suptitle("basescattering_correct (4)")
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
            simple_plot_3d(ax1, B)
            ax2.plot(x, init_ey)
            ax2.plot(x, intensity_array[:,i,1])
            ax2.plot(x, y, color='orange')
            fig.tight_layout()
            plt.show()

    return B
