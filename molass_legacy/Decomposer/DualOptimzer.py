"""
    DualOptimzer.py

    Copyright (c) 2018-2023, SAXS Team, KEK-PF
"""
import copy
import logging
import numpy as np
from scipy.optimize import minimize
from molass_legacy.Models.ElutionCurveModels import EGHA, EMGA, egha, emga
from molass_legacy._MOLASS.SerialSettings import get_setting
from DualEvaluator import DualEvaluator
import molass_legacy.KekLib.DebugPlot as plt

ADOPTABLE_LIMIT_RATIO       = 0.02
ADOPTABLE_LIMIT_RATIO_YET   = 0.03  # for minor peaks
ADOPTABLE_ADJACENT_RATIO    = 0.2   # not used
OPTIMIZABLE_LIMT_RATIO      = 0.1

class DualPairSelector:
    def __init__(self, curve, fit_recs):
        self.curve = curve
        self.peak_info = curve.peak_info
        self.boundaries = curve.boundaries
        self.fit_recs = fit_recs
        self.fit_top_ys = []
        for rec in fit_recs:
            peak = rec[3]
            y = curve.spline(peak.top_x)
            self.fit_top_ys.append(y)
        self.max_y = curve.max_y
        self.optimize_limit_y = curve.max_y * OPTIMIZABLE_LIMT_RATIO

    def select_one_pair(self):
        pair = None
        if len(self.peak_info) == 1:
            if len(self.fit_recs) == 2:
                # as in Ald
                pair = [0, 1]
        elif len(self.peak_info) == 2:
            if len(self.fit_recs) == 2:
                # as in TSsome2
                pair = [0, 1]
            elif len(self.fit_recs) == 3:
                boundary = self.boundaries[0]
                added_rec = None
                pair = []
                for k, rec in enumerate(self.fit_recs):
                    if rec[0] >= 0:
                        pair.append(k)
                    else:
                        added_rec = rec
                assert added_rec is not None
                func = added_rec[1]
                peak = added_rec[3]
                print( 'peak.top_x=', peak.top_x )
                y = func(peak.top_x)
                if self.curve.is_ignorable_peak(y):
                    pass
                else:
                    assert False
                dist = abs( peak.top_x - boundary )
                print( 'dist=', dist )
                if dist < 10:
                    pass
                else:
                    # as in OA_Ald
                    pair = None
        else:
            pass

        if pair is not None:
            assert len(pair) == 2

        return pair

    def get_nearest_major_peak_index(self, i):
        this_top_x = self.fit_recs[i][3].top_x
        min_dist = None
        nearest = None
        for k, rec in enumerate(self.fit_recs):
            if k == i or rec[0] < 0:
                continue
            dist = abs(self.fit_recs[i][3].top_x - this_top_x)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                nearest = k
        return nearest

    def is_adoptable(self, i, debug=False):

        peak_ratio = self.fit_top_ys[i]/self.max_y
        if debug:
            print([i], 'peak_ratio=', peak_ratio)

        if peak_ratio < ADOPTABLE_LIMIT_RATIO:
            if debug:
                print([i], 'is not adoptable: (1)')
            return False

        if len(self.fit_recs) == 1:
            if debug:
                print([i], 'is adoptable: (2)')
            return True

        rec = self.fit_recs[i]
        if rec[0] >= 0:
            if debug:
                print([i], 'is adoptable: (3)')
            return True

        if peak_ratio > ADOPTABLE_LIMIT_RATIO_YET:
            if debug:
                print([i], 'is adoptable: (4)')
            return True

        x = rec[3].top_x

        # get the adjacent major peak func
        k = self.get_nearest_major_peak_index(i)
        if k is None:
            if debug:
                print([i], 'is not adoptable: (5)')
            return False

        if False:
            # TODO: remove this (seems unnecessary)

            func_ = self.fit_recs[k][1]
            foot_ratio = func_(x)/self.max_y

            if debug:
                print([i,k], 'x=', x, 'foot_ratio=', foot_ratio)
            # print( [i, k], 'adjacent y=', y,  )
            # adjacent y= 9.87e-06 8.00e-06 for th 1st resid peak in Ald

            if foot_ratio > ADOPTABLE_LIMIT_RATIO**2:
                # as in the 2nd resid peak in Mar07
                if debug:
                    print([i], 'is adoptable: (6) foot_ratio=', foot_ratio)
                return True

        if debug:
            print([i], 'is not adoptable: (7)')

        return False

    def is_optimizable_rec(self, rec):
        if rec[0] >= 0:
            return True

        peak = rec[3]
        # print( 'peak.top_y=', peak.top_y, 'peak.sign=', peak.sign )
        if peak.sign < 0:
            return False

        if peak.top_y >= self.optimize_limit_y:
            return True

        return False

    def is_optimizable_pair(self, i, j, debug=False):
        assert i+1 == j or i+2 == j

        rec1 = self.fit_recs[i]
        rec2 = self.fit_recs[j]
        if not self.is_optimizable_rec(rec1) or not self.is_optimizable_rec(rec2):
            return False

        peak1 = rec1[3]
        peak2 = rec2[3]

        if j == i+1:
            # print('optimize_limit_y=', self.optimize_limit_y)
            for peak in [peak1, peak2]:
                x = peak.top_x
                y = self.curve.spline(x)
                if y < self.optimize_limit_y:
                    return False
        else:
            # as in 20160227
            pass

        if j == i+2:
            rec_ = self.fit_recs[i+1]
            func1 = rec1[1]
            func2 = rec2[1]
            func_ = rec_[1]
            peak_ = rec_[3]
            px = peak_.top_x
            py1 = func1(px)
            py_ = func_(px)
            py2 = func2(px)
            optimizable = py1/py_ > 0.5 and py2/py_ > 0.5

            if debug:
                print( 'optimizable', optimizable )
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("is_optimizable_pair debug: i=%d, j=%d" % (i, j))
                    ax.plot(self.curve.x, self.curve.y)
                    for peak in [peak1, peak2]:
                        x = peak.top_x
                        y = self.curve.spline(x)
                        ax.plot(x, y, 'o', color='red')

                    for y in [py1, py_, py2]:
                        ax.plot(px, y, 'o', color='yellow')
                    plt.show()

            return optimizable

        return True

    def compute_y_for_opt(self, x, y, i, j):
        y_ = copy.deepcopy(y)
        for k, rec in enumerate(self.fit_recs):
            if i <= k and k <= j:
                continue

            func = rec[1]
            y_ -= func(x)
        return y_

class DualOptimzer:
    def __init__(self, x, y, params1, params2, tau_hint_pair=None):
        assert type(params1) is np.ndarray and type(params2) is np.ndarray
        self.logger = logging.getLogger(__name__)
        self.x = x
        self.y = y
        self.init_params = np.concatenate([params1, params2])
        """
            editor_frame.recompute_decomposition()
                hints_dict = self.get_param_hints_dict()
                decompose_elution_better(..., hints_dict, ...)
                    decompose_elution_xray_to_uv(..., hints_dict, ...)
                        decompose_elution_impl(..., hints_dict, ...)
                            get_dual_optimized_info( ..., hints_dict  , ... )
                    decompose_elution_uv_to_xray(..., hints_dict, ...)
                        decompose_elution_impl(..., hints_dict, ...)
                            get_dual_optimized_info( ..., hints_dict  , ... )

            get_dual_optimized_info( ..., hints_dict  , ... )
                hints_dict → tau_hint_pair
                dopt = optimizer_class(..., tau_hint_pair=tau_hint_pair)
        """
        # self.tau_hint_pair = tau_hint_pair
        self.tau_bound_ratio = get_setting("TAU_BOUND_RATIO")
        # self.debug_plot("init", self.init_params)

    def debug_plot(self, title, params):
        # self.func must be given by a Subclass
        print( title, params )
        x = self.x
        y = self.y
        ax = plt.gca()
        ax.cla()
        ax.set_title(title)
        ax.plot(x,y,color='orange')
        arg1 = self.init_params[0:5]
        ax.plot(x, self.func(x,*arg1))
        arg2 = self.init_params[5:10]
        ax.plot(x, self.func(x,*arg2))
        plt.show()

    def obj_func(self, p):
        # print('obj_func: p', p)
        return np.sum(( self.func(self.x, *p[0:5]) + self.func(self.x, *p[5:10]) - self.y )**2)

    def optimize(self):
        # self.func must be given by a Subclass

        tau_bound_ratio = self.tau_bound_ratio

        #  0  1   2      3    4    5  6   7     8    9
        #  h  mu  sigma  tau  a    h  mu  sigma tau  a
        cons = [
                {'type': 'ineq', 'fun': lambda x:  x[2]*tau_bound_ratio - abs(x[3]) },  # sigma*tau_bound_ratio - abs(tau) >= 0
                {'type': 'ineq', 'fun': lambda x:  x[7]*tau_bound_ratio - abs(x[8]) },  # sigma*tau_bound_ratio - abs(tau) >= 0
            ]

        self.logger.info("minimizing with tau_bound_ratio=%g", tau_bound_ratio)
        res = minimize(self.obj_func, self.init_params, method='Nelder-Mead', constraints=cons)
        # print('opt p=', res.x)
        # self.debug_plot("optimized", res.x)
        return res

    def get_evaluators(self, res):
        # self.eval_class must be given by a Subclass
        return [ DualEvaluator(self.model, p) for p in [res.x[0:5], res.x[5:10]] ]

    def get_optrec(self, fitrec, res, k):
        rec = copy.deepcopy(fitrec)
        rec.set_evaluator(DualEvaluator(self.model, res.x[5*k:5*(k+1)]))
        return rec

    def get_avgrec(self, fitrec, res1, res2):
        rec = copy.deepcopy(fitrec)
        avg_params = (res1.x[5:10] + res2.x[0:5])/2
        rec.set_evaluator(DualEvaluator(self.model, avg_params))
        return rec

class EmgaOptimzer(DualOptimzer):
    def __init__(self, x, y, params1, params2, tau_hint_pair=None):
        self.model = EMGA()
        self.model_name = "EMGA"
        self.func = emga
        DualOptimzer.__init__(self, x, y, params1, params2, tau_hint_pair)

class EghaOptimzer(DualOptimzer):
    def __init__(self, x, y, params1, params2, tau_hint_pair=None):
        self.model = EGHA()
        self.model_name = "EGHA"
        self.func = egha
        DualOptimzer.__init__(self, x, y, params1, params2, tau_hint_pair)

def get_dual_optimized_info_old(optimizer_class, x_curve, x, y, fit_recs):
    dps = DualPairSelector(x_curve, fit_recs)
    dual_pair = dps.select_one_pair()
    print( 'dual_pair=', dual_pair )

    if dual_pair is None:
        fit_recs_ = fit_recs
    else:
        i1, i2 = dual_pair
        dopt = optimizer_class(x, y, fit_recs[i1][1].params, fit_recs[i2][1].params)
        res = dopt.optimize()
        dual_evals = dopt.get_evaluators(res)
        fit_recs_ = []
        for k, rec in enumerate([fit_recs[i1], fit_recs[i2]]):
            rec_ = copy.deepcopy(rec)
            rec_[1] = dual_evals[k]
            fit_recs_.append(rec_)

    return fit_recs_

def get_dual_optimized_info(optimizer_class, x_curve, x, y, fit_recs, hints_dict, logger, debug=False):

    # print('fit_recs=', fit_recs)

    dps = DualPairSelector(x_curve, fit_recs)
    ret_recs = []
    ign_recs = []

    n = len(fit_recs)
    pending_res = None
    i = 0
    for k in range(n):
        if i >= n:
            break

        fitrec1 = fit_recs[i]
        if hints_dict is None:
            hint1 = None
        else:
            xkey1 = fitrec1[3].get_xkey()
            hint1 = hints_dict.get(xkey1)
        if i+2 < n and dps.is_optimizable_pair(i, i+2):
            j = i+2
        elif i+1 < n and dps.is_optimizable_pair(i, i+1):
            j = i+1
        else:
            j = None

        if debug:
            from OptRecsUtils import debug_plot_opt_recs
            print('n=', n, 'i=', i, 'j=', j)
            debug_plot_opt_recs(x_curve, fit_recs, title="get_dual_optimized_info: [%d] i=%d, j=%s" % (k, i, str(j)) )

        if j is None:
            if pending_res is None:
                if dps.is_adoptable(i, debug=debug):
                    ret_recs.append(fitrec1)
                    if debug: print( [i], 'adopted' )
                else:
                    ign_recs.append(fitrec1)
                    if debug: print( [i], 'ignored' )
            else:
                rec = dopt.get_optrec(fitrec1, pending_res, 1)
                ret_recs.append(rec)
                pending_res = None
                if debug: print( [i], 'adopted pending' )
        else:
            fitrec2 = fit_recs[j]
            if hints_dict is None:
                hint2 = None
            else:
                xkey2 = fitrec2[3].get_xkey()
                hint2 = hints_dict.get(xkey2)
            y_ = dps.compute_y_for_opt(x, y, i, j)
            tau_hint_pair = ( hint1, hint2 )
            dopt = optimizer_class(x, y_, fitrec1[1].param_values, fitrec2[1].param_values, tau_hint_pair=tau_hint_pair)
            res = dopt.optimize()
            if pending_res is None:
                rec = dopt.get_optrec(fitrec1, res, 0)
                if debug: print( [i], 'adopted without pending' )
            else:
                rec = dopt.get_avgrec(fitrec1, pending_res, res)
                if debug: print( [i], 'adopted with pending' )

            ret_recs.append(rec)
            if rec[0] < 0:
                # don't let pend minor peaks
                pending_res = None
            else:
                pending_res = res

        if j is None:
            i += 1
        else:
            i = j

    logger.info( 'dual optimizer produced %d records ignoring %d records.' % ( len(ret_recs), len(ign_recs) ) )
    return ret_recs

def get_assert_info(curve, x, y, opt_recs):
    from molass_legacy.KekLib.BasicUtils import Struct
    peaks=[ rec[3].get_assert_info()  for rec in opt_recs ]

    y_ = copy.deepcopy(y)
    for rec in opt_recs:
        func = rec[1]
        y_ -= func(x)

    resid = np.abs(y_)
    navgdiff = np.sum(resid)/len(y)/curve.max_y
    nmaxdiff = np.max(resid)/curve.max_y

    return Struct( peaks=peaks, navgdiff=navgdiff, nmaxdiff=nmaxdiff )
