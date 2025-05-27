"""
    UvDiffEffects.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import os
import numpy as np
from bisect import bisect_right
from scipy.optimize import minimize, basinhopping
from molass_legacy.KekLib.SciPyCookbook import smooth
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting, get_beamline_name
from molass_legacy.Batch.StandardProcedure import StandardProcedure
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from molass_legacy.Peaks.EghSupples import egh, compute_AB, d_egh
from MatrixData import simple_plot_3d
from molass_legacy.Elution.CurveUtils import simple_plot
from DataUtils import get_in_folder
from MplAnnotate import get_annotate_position
from LPM import get_corrected

WAVELENGTHS = [280, 400]    # ng at 420 for 20170209/OAGI_01

def get_uv_data_from_file(uv_file):
    from SerialDataUtils import load_uv_array
    from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
    folder, file = os.path.split(uv_file)
    D, wv, _ = load_uv_array(folder, file)
    print(file, D.shape)
    if D.shape[1] > 100:
        i1 = bisect_right(wv, WAVELENGTHS[0])
        y1 = D[i1,:]
        ecurve = ElutionCurve(y1)
    else:
        ecurve = None
    return D, wv, ecurve

debug_eval = False

def investigate_diff_effects(in_folder, use_spline=False, fh=None, show_fig=True, fig_file=None, logger=None, uv_file=None, ret_info=False):
    global debug_eval

    if uv_file is None:
        sp = StandardProcedure()
        sd = sp.load_old_way(in_folder)

        if get_setting("use_xray_conc"):
            if logger is not None:
                logger.info("skipping %s", in_folder)
            return

        if logger is not None:
            logger.info("doing %s", in_folder)
        pre_recog = PreliminaryRecognition(sd)
        D, _, wv, ecurve = sd.get_uv_data_separate_ly()
    else:
        # uv_file is used for those without X-ray data
        sd = None
        D, wv, ecurve = get_uv_data_from_file(uv_file)
        if ecurve is None:
            return

    i1, i2 = [bisect_right(wv, v) for v in WAVELENGTHS]

    x = ecurve.x
    y = ecurve.y

    y2 = D[i2,:]

    pn = ecurve.primary_peak_no
    emg_peaks = ecurve.get_emg_peaks()
    epk = emg_peaks[pn]
    # my = egh(x, *epk.opt_params)
    # my = epk.get_model_y(x)
    h, m, s, t = epk.get_params()
    my = egh(x, h, m, s, t)

    if use_spline:
        from molass_legacy.Trimming.PeakRegion import PeakRegion
        diff_spline = ecurve.d1
        def diff_model(x, scale):
            return scale * diff_spline(x)

        if sd is None:
            if True:
                ady = np.abs(diff_spline(x))
                ady = ady/np.max(ady)
                wh = np.where(ady > 0.05)[0]
                ilp, irp = wh[[0,-1]]
            else:
                ends_list = []
                for k, epk_ in enumerate(emg_peaks):
                    h_, m_, s_, t_ = epk_.get_params()
                    print("--------------------", [k], m_)
                    # my = egh(x, h_, m_, s_, t_)
                    A, B = compute_AB(np.log(0.03), s_, t_)
                    lp, rp = m_ - A, m_ + B
                    ends_list.append((lp, rp))

                ilp, irp = [int(p) for p in [ends_list[0][0], ends_list[-1][1]]]
        else:
            from molass_legacy.UV.PlainCurve import make_secondary_e_curve_at

            xr_curve = sd.get_xray_curve()
            a_curve2 = make_secondary_e_curve_at(D, wv, ecurve, logger, wavelen=400)
            pr = PeakRegion(xr_curve, ecurve, a_curve2)
            # uv_peak_ends = pr.get_peak_ends([0.05, 0.95])[1]
            uv_peak_ends = pr.get_wider_ends()
            ilp, irp = uv_peak_ends
    else:
        def diff_model(x, scale):
            return d_egh(x, scale, m, s, t)

        A, B = compute_AB(np.log(0.05), s, t)
        lp, rp = m - A, m + B

        ilp, irp = [int(p) for p in [lp, rp]]

    pk_slice = slice(ilp, irp+1)
    print("pk_slice=", pk_slice)

    x_ = x[pk_slice]
    y2_ = y2[pk_slice]
    # sy2 = smooth(y2_, window_len=10)
    sy2 = smooth(y2_)

    def diff_func(x, scale, slope, intercept):
        return diff_model(x, scale) + slope * x + intercept

    def fit_func(p):
        return np.sum((diff_func(x_, *p) - sy2)**2)

    def evaluate(ret):
        global debug_eval
        scale = ret.x[0]
        scale_ratio = scale/h
        # dy = d_egh(x_, scale, m, s, t)
        dy = diff_model(x_, scale)
        height = np.max(dy) - np.min(dy)
        height_ratio = height/h
        print("scale_ratio=", scale_ratio)
        print("height_ratio=", height_ratio)
        rmsd = np.sqrt(np.average((diff_func(x_, *ret.x) - sy2)**2))
        print("rmsd=", rmsd)
        if debug_eval:
            slope, intercept = ret.x[1:3]
            with plt.Dp():
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
                ax1 = axes[0,0]
                ax2 = axes[1,0]
                ax1.plot(x, y)
                ymin, ymax = ax1.get_ylim()
                ax1.set_ylim(ymin, ymax)
                for j in ilp, irp:
                    ax1.plot([j, j], [ymin, ymax], color="yellow")

                # ax1.plot(x, my, ":")
                ax2.plot(x, y2)
                ax2.plot(x_, sy2)
                ax2.plot(x_, dy + x_*slope + intercept, ":", color="cyan")

                ax3 = axes[0,1]
                ax4 = axes[1,1]

                for ax, scale_ in (ax3,1), (ax4, scale):
                    ax.plot(x, diff_model(x, scale_))
                    ax.plot(x_, diff_model(x_, scale_), color="cyan")

                fig.tight_layout()
                debug_eval = plt.show()
        return scale, height, rmsd

    results = []
    # method = None
    method = "Nelder-Mead"
    for init_scale in [-1, -0.02, -0.001, 0.001,  0.02, 1]:
        ret = minimize(fit_func, (init_scale, 0, 0), method=method)
        # ret = basinhopping(fit_func, (init_scale, 0, 0))
        scale, height, rmsd = evaluate(ret)
        results.append((abs(rmsd/height), ret))

    results = sorted(results, key=lambda p: p[0])
    print("results=", [r[0] for r in results])
    opt_ret = results[0][1]
    # debug_eval = True
    scale, height, rmsd = evaluate(opt_ret)

    in_folder_ = get_in_folder(in_folder)
    uv_device_no = get_setting("uv_device_no")

    rec = [in_folder_, uv_device_no] + [str(v)  for v in [h, scale, height, rmsd]]
    if fh is None:
        print(rec)
    else:
        fh.write(",".join(rec) + "\n")

    # h_ratio = height/h
    h_ratio = height/np.max(y)
    info = uv_file, ecurve, diff_func, x, my, ilp, irp, h, m, s, t, y2, x_, sy2, opt_ret, h_ratio

    if show_fig:

        with plt.Dp():
            filename = "" if uv_file is None else "/%s" % os.path.split(uv_file)[1]
            beamline = get_beamline_name(uv_device_no)
            fig = plt.figure(figsize=(18,6))
            fig.suptitle("Differential Effects Investigation in %s%s from %s" % (in_folder_, filename, beamline), fontsize=20)
            ax1 = fig.add_subplot(131, projection="3d")
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.set_title("3D View", fontsize=16)
            ax2.set_title("Elution at λ=%d" % WAVELENGTHS[0], fontsize=16)
            ax3.set_title("Elution at λ=%d" % WAVELENGTHS[1], fontsize=16)

            simple_plot_3d(ax1, D, x=wv)
            uy = np.arange(len(y2))
            for i, c in [(i1, 'C1'), (i2, 'cyan')]:
                ux = np.ones(len(uy)) * wv[i]
                uz = D[i,:]
                ax1.plot(ux, uy, uz, color=c)

            draw_2d_figs(ax2, ax3, info)

            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            if fig_file is None:
                plt.show()
            else:
                from time import sleep
                fig.savefig(fig_file)
                plt.show(block=False)
                sleep(0.5)

    if ret_info:
        return info

def draw_2d_figs(ax2, ax3, info):
    uv_file, ecurve, diff_func, x, my, ilp, irp, h, m, s, t, y2, x_, sy2, opt_ret, h_ratio = info

    if True:
        ax2.plot(ecurve.x, ecurve.y)
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax)
        for j in ilp, irp:
            ax2.plot([j, j], [ymin, ymax], color="yellow")
    else:
        simple_plot(ax2, ecurve, legend=False)
        ax2.plot(x, my, ":", label="egh model", lw=3)
        for p in [ilp, irp]:
            ax2.plot(p, egh(np.array([p]), h, m, s, t), "o", color="yellow")
        ax2.legend()

    ax3.plot(x, y2, label="data")
    ax3.plot(x_, sy2, label="data")
    ax3.plot(x_, diff_func(x_, *opt_ret.x), ":", label="fit diff", lw=3, color="cyan")
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymin, ymax)
    for j in ilp, irp:
        ax3.plot([j, j], [ymin, ymax], color="yellow")
    ax3.legend()

    tx, ty, r = get_annotate_position(ax3, debug=False)
    ax3.text(tx, ty, "ratio=%.2g" % h_ratio, ha="center", va="center", alpha=0.5, fontsize=20)
