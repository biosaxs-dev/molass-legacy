"""

    TrimmingDebugUtils.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Elution.CurveUtils import simple_plot
from .TrimmingInfo import get_trimming_ends_from_list

def trimming_debug_plot(title, sd, uv_restrict_list, xr_restrict_list ):
    a_curve = sd.absorbance.a_curve
    D, E, qv, e_curve = sd.get_xr_data_separate_ly()
    uv_restrict_list
    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
        fig.suptitle(title)

        uv_e_ends, uv_a_ends = get_trimming_ends_from_list(uv_restrict_list)
        simple_plot(ax1, a_curve, color="blue")
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax)
        for j in uv_e_ends:
            ax1.plot([j, j], [ymin, ymax], color="yellow")

        xr_e_ends, xr_a_ends = get_trimming_ends_from_list(xr_restrict_list)
        simple_plot(ax2, e_curve, color="orange")
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax)
        for j in xr_e_ends:
            ax2.plot([j, j], [ymin, ymax], color="yellow")

        ax3.set_yscale("log")
        k = e_curve.primary_peak_i
        y = np.average(D[:,k-5:k+6], axis=1)
        ax3.plot(qv, y)
        ymin, ymax = ax3.get_ylim()
        ax3.set_ylim(ymin, ymax)
        for i in xr_a_ends:
            q = qv[i]
            ax3.plot([q, q], [ymin, ymax], color="yellow")
        fig.tight_layout()
        plt.show()

def plot_restrict_lines(ax, e_restrict, **kwargs):
    if e_restrict is None:
        return

    flag, start, stop, size = e_restrict
    if not flag:
        return

    ymin, ymax = ax.get_ylim()
    for p in [start, stop]:
        if p is None:
            continue

        ax.plot([p, p], [ymin, ymax], ':', **kwargs)

def plot_side_sigma_points(ax, restrict_info, curve, emg_peaks, sigma_point_ratio, **kwargs):
    if not restrict_info.flag:
        return

    x = curve.x
    for k, peak in enumerate(emg_peaks):
        print([k], peak.opt_params)
        y = peak.get_model_y(x)
        ax.plot(x, y, ':', linewidth=3, color='green', label='approximate model')

    sigma_min = restrict_info.start
    sigma_max = restrict_info.stop
    ymin, ymax = ax.get_ylim()
    print('plot_side_sigma_points:', sigma_min, sigma_max)
    for p in [sigma_min, sigma_max]:
        ax.plot([p, p], [ymin, ymax], label='%g-sigma boundary' % sigma_point_ratio, **kwargs)

def trimming_result_plot_impl(fig, axes, sd, curves, emg_peaks_list, old_e_restricts, ret_info, sigma_point_ratio, mpeaks, alt_info=None):
    from matplotlib.patches import Rectangle
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

    in_folder = get_in_folder()
    shapes = sd.conc_array.shape, tuple(reversed(sd.intensity_array.shape[0:2]))
    vectors = sd.lvector, sd.qvector
    uv_curve, xr_curve = curves
    uv_e_restrict, xr_e_restrict = old_e_restricts
    new_e_restricts = [ret_info[0][0], ret_info[1][0]]

    if alt_info is None:
        alt_text = ""
        shapes_ = shapes
        vectors_ = vectors
        ret_info_ = ret_info
        vnames = ("w", "q")
    else:
        alt_text = " with altering info"
        shapes_ = shapes[1], shapes[1]
        vectors_ = vectors[1], vectors[1]
        ret_info_ = [alt_info[1], ret_info[1]]
        vnames = ("q", "q")

    fig.suptitle("Trimming Info at a glance for %s%s" % (in_folder, alt_text), fontsize=20)

    for ax, title in zip(axes[0,:], ["Range Rectangle", "Elution Range", "Opposite Range"]):
        ax.set_title(title, fontsize=16)

    ret_twinxes = []
    for ax, shape, vector, info_list, vname in zip(axes[:,0], shapes_, vectors_, ret_info_, vnames):
        ax.invert_yaxis()
        e_info, a_info = info_list
        print("info_list=", info_list)
        e_max = shape[1] - 1
        a_max = shape[0] - 1
        ax.plot([0, e_max, e_max, 0, 0],
                [0, 0, a_max, a_max, 0])

        e_stt = 0 if e_info is None else e_info.start
        e_end = e_max if e_info is None else e_info.end
        a_stt = 0 if a_info is None else a_info.start
        a_end = a_max if a_info is None else a_info.end

        range_width = e_end - e_stt
        range_height = a_end - a_stt
        rect = Rectangle(
                (e_stt, a_stt), # (x,y)
                range_width,    # width
                range_height,   # height
                facecolor   = 'yellow',
                alpha       = 0.3,
            )
        ax.add_patch(rect)

        whole_width = e_max
        whole_height = a_max
        is_narrow = range_width/whole_width < 0.7

        ty = (a_stt + a_end)/2
        for k, j in enumerate([e_stt, e_end]):
            dx = whole_width * (0.5 - k) * 0.2
            if is_narrow:
                dy = whole_height * (0.5 - k) * 0.2
            else:
                dy = 0
            ax.annotate("%d" % j, xy=(j, ty+dy), xytext=(j+dx, ty+dy), ha="center", va="center", arrowprops=dict(arrowstyle="->", color='k'))

        tx = (e_stt + e_end)/2
        for k, i in enumerate([a_stt, a_end]):
            dy = whole_height * (0.5 - k) * 0.25
            ax.annotate("%s[%d]=%.3g" % (vname, i, vector[i]), xy=(tx, i), xytext=(tx, i+dy), ha="center", va="center", arrowprops=dict(arrowstyle="->", color='k'))

        ax.text(tx, ty, "shape=%s" % str(shape), ha="center", va="center")

        axt = ax.twinx()
        axt.invert_yaxis()
        axt.grid(False)
        axt.plot([0, e_max, e_max, 0, 0],
                vector[[0, 0, a_max, a_max, 0]], alpha=0)
        ret_twinxes.append(axt)

    axes_ = axes[:,1]
    ax1, ax2 = axes_
    if alt_info is None:
        simple_plot(ax1, uv_curve, color='blue', legend=False)
    else:
        simple_plot(ax1, xr_curve, color='orange', legend=False)
    simple_plot(ax2, xr_curve, color='orange', legend=False)

    for ax in axes_:
        ax.set_ylim(ax.get_ylim())

    if False:
        plot_restrict_lines(ax1, uv_e_restrict, color='gray')
        plot_restrict_lines(ax2, xr_e_restrict, color='gray')

    if alt_info is None:
        new_e_restricts_ = new_e_restricts
        curves_ = curves
        emg_peaks_list_ = emg_peaks_list
    else:
        new_e_restricts_  = [alt_info[1][0], ret_info[1][0]]
        curves_ = curves[1], curves[1]
        emg_peaks_list_ = [emg_peaks_list[1], emg_peaks_list[1]]

    for ax, restrict_info, curve, emg_peaks in zip(axes_, new_e_restricts_, curves_, emg_peaks_list_):
        plot_side_sigma_points(ax, restrict_info, curve, emg_peaks, sigma_point_ratio, color='yellow', lw=3)
        ax.legend()

    from molass_legacy.DataStructure.UvSpecCurve import UvSpecCurve
    from molass_legacy.DataStructure.XrSpecCurve import XrSpecCurve
    uv_scurve = UvSpecCurve(sd, mpeaks=mpeaks)
    xr_scurve = XrSpecCurve(sd)

    if alt_info is None:
        scurve0_tuple = (axes[0,2], None,  vectors[0], uv_scurve.get_y(), None, ret_info[0][1], shapes[0])
    else:
        scurve0_tuple = (axes[0,2], "log", vectors[1], xr_scurve.get_y(), None, alt_info[1][1], shapes[1])

    for ax, yscale, x, y, colorm, a_info_, shape in [
        scurve0_tuple,
        (axes[1,2], "log", vectors[1], xr_scurve.get_y(), None, ret_info[1][1], shapes[1]),
        ]:
        if yscale is not None:
            ax.set_yscale(yscale)
        ax.plot(x, y)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        if a_info_ is None:
            start = 0
            end = shape[0] - 1
        else:
            start = a_info_.start
            end = a_info_.end
        for j in [start, end]:
            x_ = x[j]
            ax.plot([x_, x_], [ymin, ymax], color="yellow", lw=3)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    return ret_twinxes

def trimming_result_plot(sd, curves, emg_peaks_list, old_e_restricts, ret_info, sigma_point_ratio, mpeaks, figfile=None):
    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,9))
        trimming_result_plot_impl(fig, axes, sd, curves, emg_peaks_list, old_e_restricts, ret_info, sigma_point_ratio, mpeaks)
        if figfile is None:
            plt.show()
        else:
            plt.show(block=False, pause=1)
            fig.savefig(figfile)
