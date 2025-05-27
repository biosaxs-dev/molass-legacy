"""
    EoiiCorrector.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from bisect import bisect_right
from scipy.optimize import minimize
from SvdDenoise import get_denoised_data
import molass_legacy.KekLib.DebugPlot as plt

class EoiiCorrector:
    def __init__(self, sd, paired_ranges, debug=False):
        if debug:
            from importlib import reload
            import LRF.PnoScdMap
            reload(LRF.PnoScdMap)
        from .PnoScdMap import PnoScdMap

        self.logger = logging.getLogger(__name__)
        self.logger.info("paired_ranges=%s", str(paired_ranges))
        D, E, qv, ecurve = sd.get_xr_data_separate_ly()
        self.pre_recog = sd.pre_recog
        self.D = D
        self.E = E
        self.qv = qv
        self.i = sd.xray_index
        self.x = ecurve.x
        self.y = ecurve.y
        self.paired_ranges = paired_ranges
        self.pno_map = PnoScdMap(sd, paired_ranges)

    def correct_params_list(self, func, params_list, debug=False):

        ret_params_list = []
        corrected = False
        for k, params in enumerate(params_list):
            color = self.pno_map.get_color(k)
            print([k], "color=", color)
            if color in ["yellow", "red"]:
                params = self.compute_corrected_cy(func, params, self.paired_ranges[k])
                corrected = True
            ret_params_list.append(params)

        if debug:
            x = self.x
            y = self.y

            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("EoiiCorrector correct_params_list debug")
                for ax, p_list in (ax1, params_list),(ax2, ret_params_list):
                    ax.plot(x, y)
                    for k, params in enumerate(p_list):
                        print([k], params)
                        ax.plot(x, func(x, *params), ":")
                fig.tight_layout()
                plt.show()

        return corrected, ret_params_list

    def compute_corrected_cy(self, func, params, prange, debug=False):
        f, t = prange.get_concatenated_range()
        print("prange=", prange, (f, t))
        return compute_corrected_cy_impl(self.qv, self.D, self.E, self.x, self.y, self.i, f, t, func, params, logger=self.logger, debug=debug)

USE_BOUNDED_LRF = False

def compute_corrected_cy_impl(qv, inD, inE, x, y, i, f, t, func, params, logger=None, debug=False):
    if USE_BOUNDED_LRF:
        if debug:
            from importlib import reload
            import BoundedLRF.BoundedLrfSolver
            reload(BoundedLRF.BoundedLrfSolver)
        from BoundedLRF.BoundedLrfSolver import BoundedLrfSolver
    else:
        if debug:
            from importlib import reload
            import LRF.LrfRgUtils
            reload(LRF.LrfRgUtils)
        from .LrfRgUtils import compute_rg_from_qvDEP

    slice_ = slice(f, t+1)
    x_ = x[slice_]
    target_y = func(x, *params)
    # target_y[slice_] = y[slice_]

    def objective(p):
        h, m, s, t, a, aqi, bqi = p
        c = func(x, h, m, s, t, a)
        return np.linalg.norm(aqi*c + bqi*c**2 - target_y)

    D = inD[:,slice_]
    E = inE[:,slice_]

    params_ = params

    results = []
    num_iter = 1
    for k in range(num_iter + 1):
        cy = func(x_, *params_)
        C = np.array([cy, cy**2])
        try:
            if USE_BOUNDED_LRF:
                solver = BoundedLrfSolver(qv, D, E, C=C, i=i)
                P_, C__, Rg, R_, L_, hK, hL, bq_bounds_, coerced_bq_ = solver.solve()
                aq_, bq_ = P_.T
            else:
                Cinv = np.linalg.pinv(C)
                P_ = D @ Cinv
                aq_, bq_ = P_.T
                Rg = compute_rg_from_qvDEP(qv, D, E, P_)
            results.append((cy, aq_, bq_, params_, Rg))
            if k < num_iter:
                aqi = aq_[i]
                bqi = bq_[i]
                ret = minimize(objective, (*params_, aqi, bqi))
                params_ = ret.x[0:5]
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(logger, "compute_corrected_cy: ")
            break

    if debug:
        cy, aq_, bq_, params_, Rq  = results[0]
        ig = bisect_right(qv, 1.8/Rg)
        gslice = slice(0,ig)
        gqv = qv[gslice]
        gqv2 = gqv**2
        j = np.argmax(cy)
        gy = D[gslice,j]

        with plt.Dp():
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20,5))
            fig.suptitle("EoiiCorrector compute_corrected_cy debug", fontsize=20)
            ax1.plot(x, y, color="gray", alpha=0.5)
            # ax1.plot(x_, y_)
            # ax1.plot(x, func(x, *params), ":", label="init estimation")
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim(ymin, ymax)
            for j in f, t:
                ax1.plot([j, j], [ymin, ymax], ":", color="gray")

            ax4.plot(gqv2, np.log(gy), color="gray", alpha=0.5, label="apparent data")

            for k, (cy_, aq_, bq_, params_, Rq) in enumerate(results):
                j = np.argmax(cy_)
                cj = cy_[j]

                cy = func(x, *params_)
                state_label = "init estimation" if k == 0 else "corrected"
                ax1.plot(x, cy, ":", label="[%d] %s" % (k, state_label), lw=2)

                ax2.plot(qv, aq_*cj, label="[%d] $A_q, \;(R_g=%.3g)$" % (k, Rg), alpha=0.5)
                ax3.plot(qv, bq_*cj**2, label="[%d] $B_q, \; (R_g=%.3g)$" % (k, Rg), alpha=0.5)
                ax4.plot(gqv2, np.log(aq_[gslice]*cj), label="[%d] $A_q, \;(R_g=%.3g)$" % (k, Rg), alpha=0.5)

            for ax in ax1, ax2, ax3, ax4:
                ax.legend()

            fig.tight_layout()
            plt.show()

    return params_
