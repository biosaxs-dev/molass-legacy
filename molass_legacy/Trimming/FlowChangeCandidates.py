"""
    FlowChangeCandidates.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from sklearn.cluster import KMeans
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.Elution.CurveUtils import simple_plot

END_FLATTEN_WIDTH = 5
OUTSTANDING_RATIO_LIMIT = 5     # > 3.57 for 20170304, > 4.02 for 20181212, > 4.81 for 20191114_3, > 7.22 for 20190607_1
OK_OUTSTANDING_RATIO = 2.5      # < 2.88 for 20190227_2
STD_RATIO_LIMIT = 3             # < 3.63 for 20190309_3
USE_UV_CORRECTOR = True         # can do without this in 20180602
SIMPLY_RECURSIVE = True

def get_ppk(N, k, gy):
    jj = len(gy) - N
    pp = np.argpartition(np.abs(gy), jj)
    ppN = pp[jj:]

    X = ppN.reshape((N,1))

    kp = k + 2                  # k==3 can group together near, but separate, candidates as in 20201208_3

    kmeans = KMeans(n_clusters=kp, random_state=0).fit(X)
    print("labels=", kmeans.labels_)

    pp3_ = []
    for i in range(kp):
        group = ppN[kmeans.labels_ == i]
        n = np.argmax(abs(gy[group]))
        pp3_.append(group[n])
    pp3 = sorted(pp3_, key=lambda j: -abs(gy[j]))

    return jj, pp, ppN, pp3[0:k].copy()

def get_largest_gradients(y, k, peak_region, return_full_info=False, uv_corrector=None, recursive=True, logger=None, debug=False):
    gy = np.gradient(y)
    gy[0:END_FLATTEN_WIDTH] = 0
    gy[-END_FLATTEN_WIDTH:] = 0
    peak_slice = peak_region.get_slice()
    gy[peak_slice] = 0

    N = 10
    jj, pp, ppN, pp3 = get_ppk(N, k, gy)

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        def debug_plot(title_tail=""):
            from DataUtils import get_in_folder
            in_folder = get_in_folder()
            a_curve = peak_region.a_curve
            x = a_curve.x
            with plt.Dp():
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
                fig.suptitle("Grouping Debug Plot for " + in_folder + title_tail)
                ax1.set_title("Elution Curve")
                ax2.set_title("Top %d Candidates" % N)
                ax3.set_title("%d Candidates after Grouping" % k)
                simple_plot(ax1, a_curve)
                for ax, ppk in [(ax2, ppN), (ax3, pp3)]:
                    ax.plot(x, gy, label="gradient")
                    for i, p in enumerate(ppk):
                        ax.plot(p, gy[p], "o", label=str(i))
                    ax.legend(loc="lower right")
                fig.tight_layout()
                plt.show()
        debug_plot()

    if return_full_info:
        def compute_outstanding_ratio():
            pp_ = sorted(pp[:jj])
            gy_ = gy[pp_]
            return np.max(np.abs(gy[pp3]))/np.std(gy_)

        outstanding_ratio = compute_outstanding_ratio()

        if outstanding_ratio < OUTSTANDING_RATIO_LIMIT:
            N = 15
            jj, pp, ppN, pp3 = get_ppk(N, k, gy)
            outstanding_ratio_retried = compute_outstanding_ratio()
            logger.info("retried get_ppk with N=%d to possibly improving outstanding_ratio=%.3g from %.3g", N, outstanding_ratio_retried, outstanding_ratio)
            outstanding_ratio = outstanding_ratio_retried
            if debug:
                debug_plot(" (retry)")

        smoothed = False
        corrected_y = None
        if recursive:
            outside = peak_region.get_outside()
            std_ratio = np.std(y[peak_slice])/np.std(y[outside])
            if outstanding_ratio < OUTSTANDING_RATIO_LIMIT or std_ratio > STD_RATIO_LIMIT:
                if USE_UV_CORRECTOR:
                    try:
                        uv_corrector.fit(peak_region)
                        corrected_y = uv_corrector.get_corrected_y()
                        y_for_smoothing = corrected_y
                        logger.info("elution curve for flowchange detection has been corrected due to ratios(%.3g, %.3g).", outstanding_ratio, std_ratio)
                    except:
                        log_exception(logger, "uv_corrector.fit failure: ")
                        y_for_smoothing = y
                else:
                    y_for_smoothing = y
                sy = smooth(y_for_smoothing)
                if SIMPLY_RECURSIVE:
                    sgy, spp3, sratio = get_largest_gradients(sy, k, peak_region, return_full_info=True, recursive=False, logger=logger)[0:3]
                else:
                    # may be deprecated
                    sgy, spp3_extra, sratio = get_largest_gradients(sy, k+3, peak_region, return_full_info=True, recursive=False, logger=logger)[0:3]
                    spp3 = spp3_extra[0:k]
                    spp3 = uv_corrector.get_corrected_ppn(sy, k, spp3_extra, debug=True)
                    pmax = np.max(spp3)
                    pmin = np.min(spp3)
                    pwidth = pmax - pmin
                    ppktop_x = uv_corrector.curve1.primary_peak_x
                    print("spp3[0]=", spp3[0], "ppktop_x=", ppktop_x, "pwidth=", pwidth)
                    if spp3[0] < ppktop_x and pwidth < 10:
                        # as in 20170304
                        pstart = pmax + 10
                        sgy_, spp3_ = get_largest_gradients(sy[pstart:], k, peak_region, recursive=False)
                        spp3[-1] = pstart + spp3_[0]

                if debug:
                    print("ratios = (%.3g, %.3g)" % (outstanding_ratio, sratio))
                    curve1 = uv_corrector.curve1
                    plt.push()
                    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                    fig.suptitle("get_largest_gradients recursive debug")
                    simple_plot(ax1, curve1)
                    ax2.plot(y)
                    for p in pp3:
                        ax2.plot(p, y[p], "^")
                    ax2.plot(sy, color="cyan")
                    for p in spp3:
                        ax2.plot(p, sy[p], "o")
                    fig.tight_layout()
                    plt.show()
                    plt.pop()

                if sratio > OK_OUTSTANDING_RATIO:
                    if logger is not None:
                        logger.info("largest_gradients have been replaced with smoothed ones changing o-ratio from %.3g to %.3g.", outstanding_ratio, sratio)
                    gy, pp3, outstanding_ratio = sgy, spp3, sratio
                    # pp3, outstanding_ratio = spp3, sratio
                    smoothed = True
        return gy, pp3, outstanding_ratio, smoothed, corrected_y
    else:
        return gy, pp3
