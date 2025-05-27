# coding: utf-8
"""
    Rgg.MostSimultaneous.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
import logging
from time import time
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.stats import linregress
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Peaks.ElutionModels import egh_pdf

MAPPING_PENALTY_SCALE = 1e2
VANISH_PENALTY_SCALE = 1e3
VANISH_BOUND = 0.05
TAU_LIMIT_RATIO = 0.5   # relative to sigma
ASYMMETRY_PENALTY_SCALE = 1e2
SIGMA_LIMIT_RAIO = 1.0
SIGMA_PENALTY_SCALE = 1e2
MIN_LOG10_VALUE = 1e-10
EVAL_PARAMS_DEBUG = False
USE_DLIB = False
if USE_DLIB:
    import dlib
    from molass_legacy.KekLib.BasicUtils import Struct

class SimultaneousSolver:
    def __init__(self, n_components, xr_curve, rg_curve, uv_curve, min_ratio=0.02):
        self.logger = logging.getLogger(__name__)
        self.n_components = n_components
        self.xr_curve = xr_curve
        self.rg_curve = rg_curve
        self.uv_curve = uv_curve
        self.min_ratio = min_ratio
        self.counter = -1
        if rg_curve is None:
            self.slices = None
        else:
            self.slices = rg_curve.get_valid_slices()

    def make_random_proportions(self, seed=None):
        if seed is None:
            if self.given_seeds is None:
                seed = np.random.randint(1000, 9999)
            else:
                seed = self.given_seeds[0]
        self.peaks_seed = seed
        np.random.seed(seed)
        proportions = sorted(np.random.uniform(0, 1, self.n_components))
        return proportions

    def guess_init_curve_params(self, init_proportions, curve, save_sigma=False, debug=False):
        mean, v = curve.compute_moments()
        s = np.sqrt(v)
        if save_sigma:
            self.xr_sigma = s

        x = curve.x
        y = curve.y
        max_y = curve.max_y
        xmin = max(x[0], mean - 5*s)
        xmax = min(x[-1], mean + 5*s)
        peak_topx = []
        for k, w in enumerate(init_proportions):
            x_ = xmin * (1-w) + xmax*w
            if debug:
                print([k], x_)
            peak_topx.append(x_)

        peak_topx = np.array(peak_topx)
        peak_topy= np.max([np.zeros(len(peak_topx)), curve.spline(peak_topx)], axis=0)
        w_ = np.power(peak_topy, 1/self.n_components)   # make them less distinctive
        weights = w_/np.sum(w_)

        def obj_func(heights):
            ty = np.zeros(len(y))
            for h, m, w in zip(heights, peak_topx, weights):
                ty += h * egh_pdf(x, m, s*w, 0)
            return np.sum((ty - y)**2)

        bounds = [(0, None)] * self.n_components
        try:
            result = minimize(obj_func, peak_topy, bounds=bounds)
        except Exception as exc:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "minimize")
            print(mean, s, peak_topx, peak_topy, weights)
            raise exc

        if debug:
            plt.push()
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ty = np.zeros(len(y))
            for h, m, w in zip(result.x, peak_topx, weights):
                y_ = h * egh_pdf(x, m, s*w, 0)
                ax.plot(x, y_, ':')
                ty += y_
            ax.plot(x, ty, ':', color='red')
            fig.tight_layout()
            plt.show()
            plt.pop()

        return result.x, peak_topx, s*weights

    def guess_init_rgs(self, init_proportions, rg_curve):
        x_, y_, rg_ = rg_curve.get_valid_curves()
        mean = np.mean(rg_)
        std = np.std(rg_)
        return np.linspace(mean+std, mean-std, self.n_components)

    def guess_init_mapping(self):
        xr_topx = self.xr_curve.primary_peak_x
        uv_topx = self.uv_curve.primary_peak_x
        x = np.array([self.xr_curve.x[0], xr_topx, self.xr_curve.x[-1]])
        y = np.array([self.uv_curve.x[0], uv_topx, self.uv_curve.x[-1]])
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return np.array([slope, intercept])

    def select_hopeful_init_params(self, debug=False):

        min_score = None
        min_init_params = None
        min_separate_params = None
        min_seed = None
        x = self.xr_curve.x
        init_mappable_range = (x[0], x[-1])

        k = 0
        while k < 10:
            init_proportions = self.make_random_proportions()
            try:
                init_xr_params = self.guess_init_curve_params(init_proportions, self.xr_curve, save_sigma=True)
                init_rgs = self.guess_init_rgs(init_proportions, self.rg_curve)
                init_uv_params = self.guess_init_curve_params(init_proportions, self.uv_curve)
                heights, peak_topx, sigmas = init_xr_params
                xr_params = []
                for h, m, s in zip(heights, peak_topx, sigmas):
                    xr_params.append((h, m, s, 0))
                init_mapping = self.guess_init_mapping()
                init_params = np.concatenate([np.array(xr_params).flatten(), init_rgs, init_mapping, init_uv_params[0], init_mappable_range])
                self.prepare_for_optimization(init_params)
                score = self.objective_func(init_params, debug=debug)
                print([k], "score=", score)
            except:
                continue
            if np.isnan(score):
                continue
            k += 1
            if min_score is None or score < min_score:
                min_score = score
                min_init_params = init_params
                min_separate_params = [xr_params, init_rgs, init_mapping, init_uv_params]
                min_seed = self.peaks_seed
                if self.given_seeds is not None:
                    break

        self.peaks_seed = min_seed
        return min_init_params, min_separate_params

    def solve(self, init_proportions=None, init_xr_params=None, init_rgs=None, init_mapping=None, init_uv_params=None,
                seeds=None, min_overlap=True, hopeful_only=True, debug=False):
        self.counter += 1
        self.given_seeds = seeds
        self.min_overlap = min_overlap
        self.debug_fv = False
        x = self.xr_curve.x
        init_mappable_range = (x[0], x[-1])

        if hopeful_only:
            init_params, separate_params = self.select_hopeful_init_params(debug=EVAL_PARAMS_DEBUG)
        else:
            if init_proportions is None:
                init_proportions = self.make_random_proportions()
            print("init_proportions=", init_proportions)

            if init_xr_params is None:
                init_xr_params = self.guess_init_curve_params(init_proportions, self.xr_curve, debug=debug)
            print("init_xr_params=", init_xr_params)

            if init_rgs is None:
                init_rgs = self.guess_init_rgs(init_proportions, self.rg_curve)
            print("init_rgs=", init_rgs)

            if init_uv_params is None:
                init_uv_params = self.guess_init_curve_params(init_proportions, self.uv_curve, debug=debug)
            print("init_uv_params=", init_uv_params)

            heights, peak_topx, sigmas = init_xr_params
            xr_params = []
            for h, m, s in zip(heights, peak_topx, sigmas):
                xr_params.append((h, m, s, 0))

            init_mapping = self.guess_init_mapping()
            separate_params = [xr_params, init_rgs, init_mapping, init_uv_params]
            init_params = np.concatenate([np.array(xr_params).flatten(), init_rgs, init_mapping, init_uv_params[0], init_mappable_range])

        self.prepare_for_optimization(init_params)

        if debug:
            self.objective_func(init_params, plot=True)

        if self.given_seeds is None:
            seed = np.random.randint(1000, 9999)
        else:
            seed = self.given_seeds[1]

        self.bh_seed = seed

        self.debug_fv = False       # self.debug_fv might have been set to True in other debug plots

        if USE_DLIB:
            lb, ub = self.make_bounds(separate_params)
            x, y = dlib.find_min_global(self.objective_func_for_dlib, 
                           lb,      # Lower bound constraints on x0 and x1 respectively
                           ub,      # Upper bound constraints on x0 and x1 respectively
                           10)          # T
            result = Struct(x=x, fun=y)
        else:
            result = basinhopping(self.objective_func, init_params, seed=seed)

        if debug:
            self.objective_func(result.x, plot=True)

        return result

    def get_used_seeds(self):
        return self.peaks_seed, self.bh_seed

    def split_params(self, p):
        n = self.n_components
        sep = 4*n
        xr_params = p[0:sep].reshape((n,4))
        rgs = p[sep:sep+n]
        sep = sep+n
        mapping = p[sep:sep+2]
        sep = sep+2
        uv_params = p[sep:sep+n]
        mappable_range = p[sep+n:]
        return xr_params, rgs, mapping, uv_params, mappable_range

    def prepare_for_optimization(self, init_params):
        self.xm, self.ym, self.rg = self.rg_curve.get_valid_curves()
        self.mask = self.rg_curve.get_mask()

        xr_params, rgs, mapping, uv_params, mappable_range = self.split_params(init_params)
        self.init_mapping = mapping

    def make_bounds(self, separate_params):
        xr_params, init_rgs, init_mapping, init_uv_params = separate_params
        lb = []
        ub = []
        assert False

        return lb, ub

    def objective_func_for_dlib(self, *args):
        return self.objective_func(np.array(args))

    def objective_func(self, p, debug=False, fig_info=None):
        debug_ = debug or self.debug_fv

        x = self.xr_curve.x
        y = self.xr_curve.y

        xm = self.xm
        rg = self.rg

        xr_params, rg_params, (a, b), uv_params, (c, d) = self.split_params(p)

        map_slice = slice(max(0,int(np.floor(c))), min(len(x),int(np.ceil(d))))

        uv_ty = np.zeros(len(y))
        ty = np.zeros(len(y))
        k = 0
        cy_list = []
        if debug_:
            uv_cy_list = []
        uv_x = a*x+b
        asymmetry_penalty = 0
        sigma_penalty = 0
        for h, m, s, t in xr_params:
            uv_cy = uv_params[k] * egh_pdf(uv_x, a*m+b, a*s, a*t)
            uv_ty += uv_cy
            cy = h * egh_pdf(x, m, s, t)
            cy_list.append(cy)
            if debug_:
                uv_cy_list.append(uv_cy)
            ty += cy
            asymmetry_penalty += max(0, t/s - TAU_LIMIT_RATIO)**2
            sigma_penalty += max(0, s/self.xr_sigma - SIGMA_LIMIT_RAIO)**2
            k += 1

        t_rg = np.zeros(len(rg))
        tym = ty[self.mask]
        for r, cy in zip(rg_params, cy_list):
            t_rg += r * cy[self.mask]/tym

        xr_scale = np.sum(ty**2)
        rg_scale = np.sum(t_rg**2)
        uv_scale = np.sum(uv_ty**2)

        uv_y = self.uv_curve.spline(uv_x)
        resid_xr = np.log10(np.sum((ty - y)[map_slice]**2)/xr_scale)
        resid_rg = np.log10(np.sum((t_rg -rg)**2)/rg_scale)
        resid_uv = np.log10(np.sum((uv_ty - uv_y)[map_slice]**2)/uv_scale)
        ratio = a/self.init_mapping[0]
        mapping_penalty = MAPPING_PENALTY_SCALE * ( min(0, ratio - 0.7)**2 + max(0, ratio - 1.4)**2 )
        xr_h_total = np.sum(xr_params[:,0])
        uv_h_total = np.sum(uv_params)
        vanishing_penalty = VANISH_PENALTY_SCALE * ( (min(VANISH_BOUND, np.min(xr_params[:,0])/xr_h_total) - VANISH_BOUND)**2 + (min(VANISH_BOUND, np.min(uv_params)/uv_h_total)- VANISH_BOUND)**2)
        asymmetry_penalty *= ASYMMETRY_PENALTY_SCALE
        sigma_penalty *= SIGMA_PENALTY_SCALE
        disorder_penalty = np.sum(np.min([np.zeros(xr_params.shape[0]-1), xr_params[1:,1] - xr_params[:-1,1]], axis=0)**2)
        discarded_ratio = max(MIN_LOG10_VALUE, (xr_scale - np.sum(ty[map_slice]**2))/xr_scale)
        unmapped_penalty = min(0, c)**2 + max(0, d - x[-1])**2 + max(-1.5, np.log10(discarded_ratio))

        if self.min_overlap:
            overlap = np.zeros(len(x))
            for i in range(len(cy_list)-1):
                cy1 = cy_list[i]
                cy2 = cy_list[i+1]
                overlap += np.abs(np.min([cy1, cy2], axis=0))       # np.abs() is intended to degrade negative elements
            overlap_penality = max(-1.5, np.log10(np.sum(overlap**2)/xr_scale))
        else:
            overlap_penality = 0

        scores = [max(resid_xr, resid_uv), resid_rg, mapping_penalty, vanishing_penalty, asymmetry_penalty, sigma_penalty, overlap_penality, disorder_penalty, unmapped_penalty]
        fv = np.sum(scores)

        if np.isnan(fv):
            # make clear the cause of this situation
            print("scores=", scores)
            print("xr_scale=", xr_scale)
            print("rg_scale=", rg_scale)
            print("uv_scale=", uv_scale)

        if debug_:
            from molass_legacy.Elution.CurveUtils import simple_plot

            disp_scores = [resid_xr, resid_uv, resid_rg, mapping_penalty, vanishing_penalty, asymmetry_penalty, sigma_penalty, overlap_penality, disorder_penalty, unmapped_penalty]

            dp = plt.push()
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))

            if fig_info is None:
                fig.suptitle("Simultaneous Optimization Result; fv=%.3g" % (fv), fontsize=24)
            else:
                in_folder, seeds, fv, show_scores = fig_info
                fig.suptitle("Simultaneous Optimization Result for %s with seeds=(%d, %d), fv=%.3g" % (in_folder, *seeds, fv), fontsize=24)
            ax1.set_title("UV Elution Decomposition", fontsize=20)
            ax2.set_title("Xray Elution Decomposition", fontsize=20)
            title = "Decomposition and Mapping Scores" if show_scores else "Mapped Elution Curves"
            ax3.set_title(title, fontsize=20)

            if False:
                def sort_components(vec_list):
                    sorted_list = sorted([(k, vec) for k, vec in enumerate(vec_list)], key=lambda x: xr_params[x[0],1])
                    return [rec[1] for rec in sorted_list]

                uv_cy_list = sort_components(uv_cy_list)
                cy_list = sort_components(cy_list)

            axt = ax2.twinx()
            axt.grid(False)

            simple_plot(ax1, self.uv_curve, color='blue', legend=False)
            simple_plot(ax2, self.xr_curve, color='orange', legend=False)

            k = 0
            for uv_cy, cy in zip(uv_cy_list, cy_list):
                k += 1
                ax1.plot(uv_x, uv_cy, ':', label='component-%d' % k)
                ax2.plot(x, cy, ':', label='component-%d' % k)

            ax1.plot(uv_x, uv_ty, ':', color='red', label='total')
            ax2.plot(x, ty, ':', color='red', label='total')

            axt.plot(xm, rg, ':', label='observed rg')
            axt.plot(xm, t_rg, color='gray', label='reconstructed rg')
            ymin, ymax = axt.get_ylim()
            axt.set_ylim(20, ymax)

            if self.min_overlap:
                y1 = np.zeros(len(x))
                ax2.fill_between(x, y1, overlap, fc='pink', alpha=0.2)

            if show_scores:
                y_pos = np.arange(len(disp_scores))
                ax3.barh(y_pos, disp_scores)
                ax3.set_yticks(y_pos)
                ax3.invert_yaxis() 
                ax3.set_yticklabels("resid_xr, resid_uv, resid_rg, mapping_penalty, vanishing_penalty, asymmetry_penalty, sigma_penalty, overlap_penality, disorder_penalty, unmapped_penalty".split(", "))
            else:
                ax3.plot(x, y, color='orange', label='Xray elution')
                uv_my = np.zeros(len(x))
                for cy in cy_list:
                    uv_my +=  cy/uv_ty * uv_y
                max_uv_my = np.max(uv_my)
                ax3.plot(x, uv_my/max_uv_my*self.xr_curve.max_y, ':', color='blue', label='mapped UV elution')

            def plot_mappable_range(ax, f, t):
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                for x_ in [f, t]:
                    ax.plot([x_, x_], [ymin, ymax], color='yellow')

            u, v = [a*x_ + b for x_ in [c, d]]
            plot_mappable_range(ax1, u, v)
            plot_mappable_range(ax2, c, d)

            ax1.legend(fontsize=16)
            ax2.legend(loc='upper right', fontsize=16)
            axt.legend(loc='upper left', fontsize=16)
            ax3.legend(fontsize=16)

            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            if fig_info is None:
                self.debug_fv = plt.show()
            else:
                path = self.get_figure_path()
                dp.fig.canvas.draw()
                dp.fig.savefig(path)
                self.logger.info("xr_params=%s, rg_params=%s, (a, b)=%s, uv_params=%s, (c, d)=%s", 
                                    str(xr_params), str(rg_params), str((a,b)), str(uv_params), str((c, d)))
                # plt.show()
            plt.pop()

        return fv

    def get_figure_path(self):
        from molass_legacy._MOLASS.SerialSettings import get_setting
        sub_no = -1
        while True:
            sub_no += 1
            paren = '' if sub_no == 0 else ' (%d)' % sub_no
            path  = os.path.join(get_setting("temp_folder"), "fig-%03d.jpg%s" % (self.counter, paren))
            if os.path.exists(path):
                continue
            else:
                break
        return path

    def pretend(self, seeds, result, min_overlap=True):
        self.peaks_seed, self.bh_seed = seeds
        self.prepare_for_optimization(result.x)
        mean, v = self.xr_curve.compute_moments()
        s = np.sqrt(v)
        self.xr_sigma = s
        self.min_overlap = min_overlap

def spike_demo(in_folder, num_components=4, num_trials=1, seeds=None, result=None):
    from time import time
    import logging
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Elution.CurveUtils import simple_plot
    from RgProcess.RgCurve import RgCurve
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

    logger = logging.getLogger(__name__)

    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_corrected_sd()

    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    U, wlvec, uv_curve = sd.conc_array, sd.lvector, sd.get_uv_curve()

    t0 = time()
    rg_curve = RgCurve(qv, xr_curve, D, E)
    logger.info("It took %.3g seconds for rg_curve construction.", time()-t0)
    segments = rg_curve.get_curve_segments()

    solver = SimultaneousSolver(num_components, xr_curve, rg_curve, uv_curve)
    in_folder_ = get_in_folder(in_folder)

    init_seeds = np.random.randint(100000, 999999, size=num_trials)
    for k, init_seed in enumerate(init_seeds):
        logger.info("RandomState init_seed=%d", init_seed)
        np.random.RandomState(seed=init_seed)
        t0 = time()
        logger.info("solving with given seeds=%s.", str(seeds))
        if result is None:
            result = solver.solve(seeds=seeds, debug=EVAL_PARAMS_DEBUG)
        else:
            solver.pretend(seeds, result)
        used_seeds = solver.get_used_seeds()
        solver.objective_func(result.x, plot=True, fig_info=[in_folder_, used_seeds, result.fun, False])
        logger.info("It took %.3g seconds in Basin Hopping optimization with seeds=(%d, %d), fv=%.3g.", time()-t0, *used_seeds, result.fun)
        seeds = None
        result = None
