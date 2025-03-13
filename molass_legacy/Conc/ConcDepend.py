"""
    ConcDepend.py

    Copyright (c) 2020-2025, SAXS Team, KEK-PF
"""
import logging
from bisect import bisect_right
import numpy as np
from scipy.optimize import minimize
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from molass_legacy._MOLASS.SerialSettings import get_xray_picking

LIMIT_Q = 0.2
CD_COLORS = ['green', 'yellow', 'red']
# CD_BOUNDARIES = [0.5, 1]
CD_BOUNDARIES = [2, 5]
BOUNDED_SCALING = True
USE_SMOOTHED_AQ_SCALED_BQ = True
RDR_COMPUTE_LIMIT_RATIO = 0.5
RANK2_RDR_LIMIT = 0.02  # < 0.02xx for 20190527 (EndoAB)
RANK2_SCD_BY_RDR = 3    # this makes the color to be 'yellow'

def compute_min_norm_scaled(y1, y2):
    def obj_func(p):
        return np.linalg.norm(y1-p[0]*y2)**2
    result = minimize(obj_func, np.ones(1))
    scale = result.x[0]
    return scale*y2

def compute_min_norm_scaled_wide(y1, y2, wslice):
    y1_ = y1[wslice]
    y2_ = y2[wslice]
    def obj_func(p):
        return np.linalg.norm(y1_-p[0]*y2_)**2
    result = minimize(obj_func, np.ones(1))
    scale = result.x[0]
    print('scale=', scale)
    return scale*y2

def compute_min_norm_bq_s_aq(y1, y2, init_scale, logger=None, ret_no_bnd=False, ret_scale=False):
    def obj_func(p):
        return np.linalg.norm(y1-p[0]*y2)**2

    # bnd = init_scale*2
    # result = minimize(obj_func, (init_scale,), bounds=((-bnd, +bnd),))
    result = minimize(obj_func, (init_scale,))
    scale = result.x[0]
    if BOUNDED_SCALING:
        ratio = abs(scale/init_scale)
        if logger is not None:
            logger.info('scale ratio: %.3g/%.3g=%.3g', scale, init_scale, ratio)
        sy2_no_bnd = None
        if ratio > 3 and False:
            # as for 20180526/OA
            sy2_no_bnd = scale*y2
            scale = 0
            if logger is not None:
                logger.info('scale is set to zero due to too large ratio %.3g', ratio)
        if ret_no_bnd:
            return scale*y2, sy2_no_bnd

    if ret_scale:
        return scale*y2, scale
    else:
        return scale*y2

def compute_distinct_cd(M, C, cd_slice, scale, logger):
    from molass_legacy.QuickAnalysis.JudgeHolder import convert_to_degrees
    try:
        U, sigmas, VT = np.linalg.svd(M)
        rank = 2
        M_ = U[:,0:rank] @ np.diag(sigmas[0:rank]) @ VT[0:rank,:]

        c = C[0,:]
        c = c/np.max(c)
        C_ = np.array([c, c**2])
        P = M_ @ np.linalg.pinv(C_)
        a = smooth(P[:,0])
        b = smooth(P[:,1])

        init_scale = sigmas[1]/sigmas[0]
        sa = compute_min_norm_bq_s_aq(b, a, init_scale, logger)
        y = b - sa
        cd_score = np.sqrt(np.average(y[cd_slice]**2))*100/scale
        cd = convert_to_degrees([cd_score])

        logger.info("cd=%d from cd_score:%.3g", cd[0], cd_score)
        ret_cd = cd[0]
    except:
        # as in 20201122
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(logger, "compute_distinct_cd failed: ")
        ret_cd = 1
        logger.warning("cd=%d due to the last error.", ret_cd)

    return ret_cd

class ConcDepend:
    def __init__(self, q, data, error, x_curve, xray_scale=None):
        self.logger = logging.getLogger(__name__)
        self.q = q
        self.xray_picking = get_xray_picking()
        self.i = bisect_right(q, self.xray_picking)
        self.ip = bisect_right(q, self.xray_picking+0.01)
        self.limit = bisect_right(q, LIMIT_Q)
        self.data = data
        self.error = error
        self.x_curve = x_curve
        self.xray_scale = xray_scale
        self.rdr_hints = None

    def compute_judge_info(self, rdr_hints, debug_info=None):
        x_curve = self.x_curve
        aslice = slice(0, self.limit)
        scd_list = []
        rdr_list = []
        islice = slice(self.i, self.ip)
        x = self.x_curve.x
        max_y = self.x_curve.max_y

        if len(x_curve.peak_info) == len(rdr_hints):
            # as in most usual cases
            proc_peak_info = x_curve.peak_info
        else:
            # as in 20170506/ACTM002/Backsub with an ecurve other than x_curve
            # this may have been fixed with x_curve, i.e., no need of this else clause
            proc_peak_info = x_curve.get_major_peak_info()

        if debug_info is not None:
            if len(rdr_hints) != len(proc_peak_info):
                # as in 
                from OnTheFly.DebugDialog import DebugDialog
                dialog = DebugDialog(debug_info=[rdr_hints, x_curve, debug_info[0]])
                dialog.show()

        for k, peak_rec in enumerate(proc_peak_info):
            print([k], peak_rec)
            start = peak_rec[0]
            stop = peak_rec[2]+1
            if start >= stop:
                # as in 20171226
                # or should be investigated for the cause
                continue

            eslice = slice(start, stop)
            M = self.data[:,eslice]
            E = self.error[:,eslice]
            U, s, VT= np.linalg.svd(M)
            print('s[0:2]=', s[0:2])
            M1_ = get_denoised_data(M, rank=1)
            M2_ = get_denoised_data(M, rank=2)
            c = x_curve.y[eslice]
            c = c/np.max(c)
            C1 = np.array([c])
            C2 = np.array([c, c**2])
            C1inv = np.linalg.pinv(C1)
            C2inv = np.linalg.pinv(C2)
            P1 = np.dot(M1_, C1inv)
            P2 = np.dot(M2_, C2inv)
            A1 = P1[:,0]
            A2 = P2[:,0]
            B2 = P2[:,1]

            if self.xray_scale is None:
                s1 = np.average(A1[islice])
                s2 = np.average(A2[islice])
                scale = np.sqrt(s1*s2)
            else:
                scale = self.xray_scale

            if USE_SMOOTHED_AQ_SCALED_BQ:
                init_scale = s[1]/s[0]
                B2_ = smooth(B2)
                A2_ = smooth(A2)
                sy2 = compute_min_norm_bq_s_aq(B2_, A2_, init_scale)
                y = B2_ - sy2
            else:
                sy2 = compute_min_norm_scaled(A2, B2)
                y = A2 - sy2
            scd = np.sqrt(np.average(y[aslice]**2))*100/scale
            # bqn = np.sqrt(np.average(B2[aslice]**2))*100/scale
            # print([k], 'scd, bqn=', scd, bqn)
            top_x = peak_rec[1]

            computable, rdr = rdr_hints[k]
            if not computable:
                # to avoid replacing the scd
                rdr = 0

            height_ratio = self.x_curve.spline(top_x)/max_y
            if height_ratio < RDR_COMPUTE_LIMIT_RATIO or scd >= CD_BOUNDARIES[0]:
                # keep the scd as is
                pass
            else:
                # rdr = self.compute_rdr_legacy(M, E, P1, A1, A2)
                if abs(rdr) > RANK2_RDR_LIMIT:
                    self.logger.info("[%d] scd %.3g has been changed to %.3g due to rdr %.3g", k, scd, RANK2_SCD_BY_RDR, rdr)
                    scd = RANK2_SCD_BY_RDR

            scd_list.append((top_x, scd))
        return scd_list

    def compute_rdr_legacy(self, M, E, P1, A1, A2):
        from molass_legacy.QuickAnalysis.RgDiffRatios import compute_rdr_legacy_impl
        return compute_rdr_legacy_impl(self.q, M, E, P1, A1, A2)
