"""
    FlowChangeUtils.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import curve_fit
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.KekLib.ScipyUtils import get_spline
from .Sigmoid import sigmoid, bent_sigmoid, fit_bent_sigmoid, ex_sigmoid
from .FlowChangeCandidates import OUTSTANDING_RATIO_LIMIT

COV_LIMIT = 1e2
L_RATIO_IGMORE_LIMIT = 0.14     # > 0.137 for 20160227, < 0.208 for fc1 in Sugiyama, > 0.13 for 20181212, < 0.144 for 20180316
FIT_HALF_WIDTH = 100
OK_FIT_BALANCE  = 0.2           # > 0.099 for 20170304
OK_K_LIMIT  = 0.1               # < 0.141 for 20170304
RETRY_K_LIMIT = 0.09            # < 0.0975 for 
SAFE_L_RATIO = 0.2              # < 0.27 for Sugiyama
SAFE_FIT_BALANCE = 0.7          # < 0.893 for OA, > 0.673 for , , < 0.95 for OA_Ald
SAFE_DIST_RATIO = 1.0           # > 0.211 for 20190309_1
SAFE_FWD_L_RATIO = 0.25         # > 0.021 for 20180526_OA
SAFE_K_LIMIT = 0.25             # < 0.141 for 20170304, < 0.261 for 20190305_1, < 3.38 for 20161216, < 0.861 for Kosugi3a, < 0.315 for 20170304
SAFE_ERROR_RATIO = 0.7          # > 0.269 for 20161216, > 0.398 for Sugiyama, < 3.83 for 20181127
MUST_ERROR_RATIO = 2            # 
MUST_FIT_BALANCE = 10           # < 13.9 for 20181127, > 9.4 for Sugiyama
FWD_LENGTH_LIMIT = 1.0
OK_TOTAL_ERROR_RATIO = 2.0      # < 2.24 for protains5

def is_ignorable_L_ratio(L_ratio):
    return L_ratio < L_RATIO_IGMORE_LIMIT

def fit_sigmoid_impl(x_, y_, height, peak_region, std_p, ppk, outstanding_ratio, logger, debug=False):
    assert len(x_) > 0

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        print("------- ppk=", ppk)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("fit_sigmoid_impl entry")
            ax.plot(x_, y_)
            for i, p in enumerate(ppk):
                ax.plot(p, y_[p - x_[0]], "o", label=str(i))
            ax.legend()
            plt.show()

    ret_popt = None
    ret_error_ratio = None
    for i, p in enumerate(ppk):
        if debug:
            print([i, p], "---------------------- try")
        try:
            popt, error_ratio = fit_sigmoid_impl_impl(x_, y_, height, peak_region, p, std_p, outstanding_ratio, logger, debug=debug)
        except Exception as exc:
            # to make it robust
            if debug:
                print([i, p], "---------------------- except", exc)
                log_exception(logger, "fit_sigmoid_impl_impl failure: ")
            continue

        L = popt[0]
        L_ratio = abs(L)/height
        if debug:
            print([i, p], "---------------------- error_ratio=%.3g, L_ratio=%.3g" % (error_ratio, L_ratio))
        if ret_error_ratio is None or error_ratio < ret_error_ratio:
            ret_popt = popt
            ret_error_ratio = error_ratio

        if not is_ignorable_L_ratio(L_ratio):
            break

    if i > 0:
        logger.info("fit_sigmoid_impl tried %d times.", i+1)

    return ret_popt, ret_error_ratio

def fit_sigmoid_impl_impl(x_, y_, height, peak_region, x0, std_p, outstanding_ratio, logger, narrower=False, debug=False):
    j0 = x0 - x_[0]
    half_width = FIT_HALF_WIDTH
    if narrower:
        half_width //= 2
    j_start = max(0, j0 - half_width)
    j_end = min(len(y_) - 1, j0 + half_width)
    try:
        y_lower = y_[j_start]
        y_upper = y_[j_end]
    except:
        print((j_start, j_end))
        assert False
    L0 = y_upper - y_lower
    p0 = L0, x0, 1, y_lower

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            print("narrower=", narrower)
            fig, ax = plt.subplots()
            ax.set_title("fit_sigmoid_impl debug (1)")
            ax.plot(x_, y_)
            ax.plot(x_, sigmoid(x_, *p0))
            fig.tight_layout()
            plt.show()

    fit_slice = slice(j_start, j_end+1)
    fx = x_[fit_slice]
    fy = y_[fit_slice]
    sigmoid_ok = False
    try:
        popt, pcov = curve_fit(sigmoid, fx, fy, p0)
        pcovd = np.diag(pcov)
        if np.isfinite(pcovd[0]):
            error_ratio = abs(np.std(sigmoid(x_, *popt) - y_)/popt[0])
        else:
            # when warned "Covariance of the parameters could not be estimated"
            error_ratio = np.inf

        if False:
            print("popt=", popt)
            print("pcov=", pcovd)
            print("error_ratio=", error_ratio)
            print("std_p=", std_p)
            # assert pcovd[0] < COV_LIMIT
        error_ratio_str = "error_ratio=%.3g" % error_ratio
        sigmoid_ok = True
    except Exception as exc:
        log_exception(logger, "curve_fit(sigmoid, fx, fy, p0) failure: ")
        error_ratio = None
        error_ratio_str = str(exc)

    bent_sigmoid_ok = False
    if (error_ratio is None
        or error_ratio > 0.2                    # error_ratio=0.228 for 20200304_1
        or error_ratio > 0.05 and std_p < 5     # for (*) below
        # (*) as in 20161216 where error_ratio=0.86, or SUB_TRN1 where error_ratio=0.059
        ):
        logger.info('trying bent_sigmoid due to "%s"', error_ratio_str)
        try:
            ret_popt, pcov = fit_bent_sigmoid(fx, fy, x0)
            if np.isfinite(pcov[0,0]):
                error_ratio = abs(np.std(bent_sigmoid(fx, *ret_popt) - fy)/ret_popt[0])
            else:
                error_ratio = np.inf
            s1, s2 = ret_popt[4:]
            print("fit_sigmoid_impl_impl: (s1, s2)=(%.3g, %.3g)" % (s1, s2))
            bent_sigmoid_ok = True
        except:
            log_exception(logger, "fit_bent_sigmoid failure: ")

    if bent_sigmoid_ok:
        pass
    else:
        if sigmoid_ok:
            ret_popt = np.zeros(6)
            ret_popt[0:4] = popt
        else:
            raise Exception("No fits at x0=%.4g" % (x0))

    L, x0, k = ret_popt[0:3]
    dist_ratio = peak_region.get_distance_ratio(x0)
    fwd_slice = slice(int(round(x0 - fx[0])), None)
    ffx = fx[fwd_slice]
    ffy = fy[fwd_slice]
    try:
        spline = get_spline(ffx, ffy)
        sffy = spline(ffx)
        fity = ex_sigmoid(ffx, *ret_popt)
        fwd_L_ratio = np.std(sffy - fity)/abs(L)
    except:
        log_exception(logger, "get_spline: ")
        # as in Sugiyama
        sffy = ffy          # for debug
        fwd_L_ratio = 0     # to avoid to be discarded

    fit_balance = abs((x0 - fx[0])/(fx[-1] - x0) - 1)

    total_error_ratio = abs(np.std(ex_sigmoid(x_, *ret_popt) - y_)/ret_popt[0])

    if debug:
        print("narrower=", narrower)
        print("error_ratio=%.3g, fwd_L_ratio=%.3g" % (error_ratio, fwd_L_ratio))
        print("total_error_ratio=%.3g" % total_error_ratio)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("fit_sigmoid_impl debug (2)")
            ax.plot(x_, y_)
            ax.plot(x_, ex_sigmoid(x_, *ret_popt))
            ax.plot(ffx, sffy)
            ax.plot(ffx, ex_sigmoid(ffx, *ret_popt))
            tx = np.average(ax.get_xlim())
            ty = np.average(ax.get_ylim())
            ax.text(tx, ty, "fit_balance=%.3g, k=%.3g, error_ratio=%.3g, fwd_L_ratio=%.3g" % (fit_balance, k, error_ratio, fwd_L_ratio), ha="center", alpha=0.5, fontsize=16)
            fig.tight_layout()
            plt.show()

    reasons = ""
    def add_reason(reason):
        nonlocal reasons
        if reasons == "":
            reasons += "reasons"
        reasons += reason

    if total_error_ratio > OK_TOTAL_ERROR_RATIO:
        add_reason("(-1) total_error_ratio=%.3g > %.3g" % (total_error_ratio, OK_TOTAL_ERROR_RATIO))

    L_ratio = abs(L)/height

    if error_ratio > MUST_ERROR_RATIO or fit_balance > MUST_FIT_BALANCE:
        add_reason(" (0) error_ratio=%.3g > %.3g or fit_balance=%.3g > %.3g" % (error_ratio, MUST_ERROR_RATIO, fit_balance, MUST_FIT_BALANCE))

    if k > SAFE_K_LIMIT:
        if error_ratio > SAFE_ERROR_RATIO:
            add_reason(" (1) error_ratio=%.3g > %.3g" % (error_ratio, SAFE_ERROR_RATIO))
        if L_ratio > SAFE_L_RATIO:
            # as in Sugiyama
            pass
        else:
            if fit_balance > SAFE_FIT_BALANCE:
                # as in OA_Ald right
                print("L_ratio=", L_ratio)
                add_reason(" (2) fit_balance=%.3g > %.3g" % (fit_balance, SAFE_FIT_BALANCE))
    else:
        if (k > OK_K_LIMIT
            and fit_balance < OK_FIT_BALANCE
            and outstanding_ratio > OUTSTANDING_RATIO_LIMIT     # required to exclude 20200211
            ):
            # as in retry in 20170304
            pass
        else:
            if not narrower and k > RETRY_K_LIMIT:
                # as in 20190305_1
                try:
                    p = int(round(x0))
                    ret_popt, error_ratio = fit_sigmoid_impl_impl(x_, y_, height, peak_region, p, std_p, outstanding_ratio, logger, narrower=True, debug=False)
                    logger.info("retry successful with k=%.3g", k)
                except Exception as exc:
                    raise exc
            else:
                add_reason(" (3) k=%.3g < %.3g" % (k, SAFE_K_LIMIT))
                if fit_balance > SAFE_FIT_BALANCE:
                    add_reason(" (4) fit_balance=%.3g > %.3g" % (fit_balance, SAFE_FIT_BALANCE))
                elif dist_ratio < SAFE_DIST_RATIO and fwd_L_ratio > SAFE_FWD_L_RATIO:
                    add_reason(" (5) dist_ratio=%.3g < =%.3g " % (dist_ratio, SAFE_DIST_RATIO))

    if fwd_L_ratio > SAFE_FWD_L_RATIO:
        add_reason(" (6) fwd_L_ratio=%.3g > %.3g" % (fwd_L_ratio, SAFE_FWD_L_RATIO))

    if reasons != "":
        logger.info("candidate at %.4g will be discarded due to %s", x0, reasons)
        raise Exception("Bad fit at x0=%.4g: %s" % (x0, reasons))

    return ret_popt, error_ratio
