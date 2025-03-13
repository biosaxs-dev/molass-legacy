"""
    Theory.Rg.py

    Copyright (c) 2020-2021, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy.stats import linregress
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from .SolidSphere import get_boundary_params_simple

GQ_LIMIT = 0.05

def compute_Rg(qv, I):
    x = qv**2
    y = np.log(I)
    slope, intercept = linregress(x, y)[0:2]
    Rg = np.sqrt(-3*slope)
    return Rg

def compute_corrected_Rg_impl(qv, M, E, initRg, gf, gt):
    x = qv[gf:gt]**2
    b1 = get_boundary_params_simple(initRg)[0]
    i = min(gt-1, bisect_right(qv, b1))
    c = M[i,:]
    C = np.array([c, c**2])
    P = M[0:gt,:] @ np.linalg.pinv(C)
    Aq = P[:,0]
    y = np.log(Aq[gf:])

    slope, intercept = linregress(x, y)[0:2]
    Rg = np.sqrt(-3*slope)

    if False:
        print('initRg=', initRg)
        print('Rg=', Rg)
        plt.push()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, x*slope+intercept)
        fig.tight_layout()
        plt.show()
        plt.pop()

    return Rg

def compute_corrected_Rg(sd, ecurve, pno, qv, M, E, range_=None):
    if pno == ecurve.primary_peak_no:
        gf, gt_ = sd.pre_recog.get_gunier_interval()
        preRg = sd.pre_recog.get_rg()
    else:
        try:
            pre_rg = sd.pre_recog.pre_rg
            if range_ is None:
                # note that pno here can be buggy!
                selected = ecurve.peak_info[pno][1]
            else:
                selected = ecurve.get_peak_position(*range_)
            preRg = pre_rg.compute_rg(selected=selected)
            sg = pre_rg.sg
            gf = sg.guinier_start
            gt_ = sg.guinier_stop
        except:
            log_exception(None, "compute_corrected_Rg (1): ")

    print('qv[gt_]=', qv[gt_])
    if qv[gt_] > GQ_LIMIT:
        gt = bisect_right(qv, GQ_LIMIT)
        guinier_truncated = ' (truncated by q <= %g)' % GQ_LIMIT
    else:
        gt = gt_
        guinier_truncated = ''
    print((gf, gt), 'Rg=%.3g' % preRg)
    try:
        Rg = compute_corrected_Rg_impl(qv, M, E, preRg, gf, gt)
    except:
        # as in Kosugi8_Backsub
        log_exception(None, "compute_corrected_Rg (2): ")
        Rg = preRg

    return Rg, gf, gt, guinier_truncated
