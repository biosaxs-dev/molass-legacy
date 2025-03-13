"""
    UnifiedDecompResult.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import copy
import numpy as np
from molass_legacy.KekLib.BasicUtils import Struct
import molass_legacy.KekLib.DebugPlot as plt
from OptRecsUtils import compute_area_proportions

SUFFICIENT_TOP_RATIO    = 0.3
SELECTABLE_TOP_RATIO    = 0.085 # < 0.086 for 20161202,  > 0.082 for OAGIwyatt_02, < 0.11 for OA_Ald
DEPENDENT_DIFF_RATIO    = 0.02  # > 0.006 for the 1st minor peak of 20181203, > 0.034 for Dec02, < 0.026 for Dec02
DEPENDENT_MAJOR_RATIO   = 0.5   # < 0.89 for the 1st minor peak of 20181203, < 0.57 for monoE
DEPENDENT_LARGER_RATIO  = 0.25  # < 0.28 for OA_Ald
FWHM_SPAN_RATIO         = 1.3   # < 1.34 for Mar01
AREA_PROPORTION_LIMIT   = 0.03  # > 0.0096 for 2-th peak HasA(EGH)
IGNORABLE_PROPORTION    = 0.03

class UnifiedDecompResult:
    def __init__(self,
                    xray_to_uv=None,
                    x_curve=None, x=None, y=None,
                    opt_recs=None,
                    max_y_xray = None,
                    model_name=None,
                    decomposer=None,
                    uv_y=None,
                    opt_recs_uv=None,
                    max_y_uv = None,
                    nresid_uv=None,
                    global_flag=False,
                    debug_info=None,
                    ): 
        self.xray_to_uv = xray_to_uv
        self.x_curve = x_curve
        self.x = x
        self.y = y
        self.opt_recs = opt_recs
        self.max_y_xray = max_y_xray
        self.model_name = model_name
        self.decomposer = decomposer
        self.uv_y = uv_y
        self.opt_recs_uv = opt_recs_uv
        self.max_y_uv = max_y_uv
        self.debug_info = debug_info
        self.nresid_uv = None if uv_y is None else self.compute_normalized_resid()
        self.global_flag = global_flag
        self.marked_as_dependent = np.zeros(len(opt_recs))

    def compute_normalized_resid(self):
        resid_y = copy.deepcopy(self.uv_y)

        for rec in self.opt_recs_uv:
            func = rec[1]
            resid_y -= func(self.x)

        nresid = np.average(np.abs(resid_y)) / self.max_y_uv

        if self.debug_info is not None:
            print("nresid=", nresid)
            x = self.x
            uv_y = self.uv_y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(self.debug_info[0])
                ax.plot(x, uv_y)
                for rec in self.opt_recs_uv:
                    func = rec[1]
                    cy = func(x)
                    ax.plot(x, cy, ":")
                ax.plot(x, resid_y, ":", color="red")
                fig.tight_layout()
                plt.show()

        return nresid

    def set_area_proportions(self):
        def set_area_proportions_impl(x, recs):
            areas = []
            for k, rec in enumerate(recs):
                cy = rec.evaluator(x)
                areas.append(np.sum(cy))
            props = np.array(areas)/np.sum(areas)
            for rec, p in zip(recs, props):
                rec.peak.area_prop = p

        set_area_proportions_impl(self.x, self.opt_recs)
        set_area_proportions_impl(self.x, self.opt_recs_uv)

    def get_range_edit_info(self, logger=None, debug=False):
        if debug:
            from importlib import reload
            import Decomposer.OptRecsUtils
            reload(Decomposer.OptRecsUtils)

        opt_recs = self.opt_recs
        x = self.x

        if debug:
            from molass_legacy.Decomposer.OptRecsUtils import debug_plot_opt_recs
            debug_plot_opt_recs(self.x_curve, opt_recs, eval_x=x, title="get_range_edit_info entry opt_recs")

        select_matrix = []
        editor_ranges = []
        top_x_list = []

        num_eltns = len(opt_recs)
        size_x = len(x)
        for k, rec in enumerate(opt_recs):
            ee_selection = [0] * num_eltns
            ee_selection[k] = 1
            select_matrix.append(ee_selection)

            range_list = rec.get_range_list(x)

            editor_ranges.append(range_list)
            peak = rec[3]

            # - self.x[0] is needed for Stochastic Models while self.x[0] == 0 for traditional models
            top_x_list.append( peak.top_x - self.x[0] )

        if debug:
            print('select_matrix(before update)=', select_matrix)

        self.update_all_selections(select_matrix, editor_ranges, top_x_list, logger, debug=debug)

        if debug:
            from molass_legacy.Decomposer.OptRecsUtils import debug_plot_opt_recs_impl
            print('select_matrix(after update)=', select_matrix)
            y = self.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_range_edit_info exit opt_recs with editor_ranges")
                debug_plot_opt_recs_impl(ax, x, y, opt_recs)
                for range_list in editor_ranges:
                    for f, t in range_list:
                        ax.axvspan(x[f], x[t], alpha=0.3, color="cyan")
                fig.tight_layout()
                plt.show()

        return Struct(
                    select_matrix=select_matrix,
                    editor_ranges=editor_ranges,
                    top_x_list=top_x_list,
                    )

    def update_all_selections(self, select_matrix, editor_ranges, top_x_list, logger, debug=False):
        opt_recs = self.opt_recs
        max_y = self.max_y_xray

        for k, rec in enumerate(opt_recs):
            if self.is_dependent_peak(k, top_x_list, logger, debug):
                if debug:
                    print([k], 'updating for the dependent peak')
                self.update_one_selection(0, select_matrix, k)
                editor_ranges[k].clear()
                self.marked_as_dependent[k] = 1

    def update_one_selection(self, num_ranges, select_matrix, k, ignorable_flags=None):
        nearest_k = self.get_nearest_major_peak_index(k, ignorable_flags)
        if nearest_k is None:
            # TODO: pH6
            return

        major_selection = select_matrix[nearest_k]
        select_list = select_matrix[k]
        if num_ranges == 0:
            select_list[k] = 0
            major_selection[k] = 1
        else:
            select_list[k] = 1
            major_selection[k] = 0

    def get_nearest_major_peak_index(self, this_k, ignorable_flags=None):
        opt_recs = self.opt_recs

        nearest_k = None
        nearest_diff = None
        this_x = opt_recs[this_k][3].top_x
        for k, rec in enumerate(opt_recs):
            if k == this_k or rec[0] < 0 or (ignorable_flags is not None and ignorable_flags[k] == 1):
                continue
            other_x = rec[3].top_x
            diff = abs(this_x - other_x)
            if nearest_diff is None or diff < nearest_diff:
                nearest_diff = diff
                nearest_k = k
        return nearest_k

    def is_dependent_peak(self, k, top_x_list, logger, debug=False):
        opt_recs = self.opt_recs
        max_y = self.max_y_xray

        rec = opt_recs[k]
        rec_id = rec[0]
        if rec_id >= 0:
            # major peak should be indepenent
            return False

        minor_rec = rec
        peak = minor_rec[3]
        if peak.sign == -1:
            # negative peak must be dependent
            return True

        top_x = top_x_list[k]
        func = minor_rec[1]
        top_y = func(top_x)
        top_ratio = top_y/max_y
        if debug:
            print( [k], 'top_ratio=', top_ratio, 'top_y=', top_y, 'max_y=', max_y )
        top_dependence = top_ratio < SELECTABLE_TOP_RATIO

        minor_peak = minor_rec[3]
        minor_func = minor_rec[1]
        major_k = self.get_nearest_major_peak_index(k)
        if major_k is None:
            # occured in pH6
            return False

        major_rec = opt_recs[major_k]
        major_peak = major_rec[3]
        major_top_x = major_peak.top_x
        top_dist = abs(top_x - major_top_x)
        span_ratio = top_dist/major_peak.fwhm
        if debug:
            print( [k], 'top_dist=', top_dist, 'major_peak.fwhm=', major_peak.fwhm, 'span_ratio=', span_ratio )
        if span_ratio > FWHM_SPAN_RATIO:
            # too distant peak cannot be dependent
            return False

        major_func = major_rec[1]
        # print('minor_func=', minor_func)
        # print('minor_func.dict_params=', minor_func.dict_params)
        x = np.arange(minor_peak.flimL, minor_peak.flimR+1)
        y1 = major_func(x)
        y2 = minor_func(x)
        upper_y = np.max([y1, y2], axis=0)
        total_y = y1 + y2
        diff_y = np.abs(total_y - upper_y)
        denom = np.sum(upper_y)
        diff_ratio = abs(np.sum(diff_y)/denom)
        major_ratio = abs(np.sum(y1)/denom)
        sub_y = y2 - y1
        larger_ratio = np.sum(sub_y[sub_y > 0])/abs(np.sum(y2))

        if debug:
            print('top_x=', top_x, 'max_y=', max_y, 'top_ratio=', top_ratio)
            print('diff_ratio=', diff_ratio, 'major_ratio=', major_ratio)
            print('larger_ratio=', larger_ratio)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(str(minor_peak))
                ax.plot(x, total_y, label='total_y', linewidth=3, alpha=0.3)
                ax.plot(x, y1, label='major_y', linewidth=3, alpha=0.3)
                ax.plot(x, y2, label='minor_y', linewidth=3, alpha=0.3)
                ax.plot(x, upper_y, ':', label='upper_y', linewidth=3)
                ax.plot(x, diff_y, ':', label='diff_y', linewidth=3)
                ax.legend()
                fig.tight_layout()
                plt.show()

        if larger_ratio > DEPENDENT_LARGER_RATIO:
            return False

        # check these ratios before checking the top_ratio
        shape_dependence = ( diff_ratio < DEPENDENT_DIFF_RATIO and major_ratio > DEPENDENT_MAJOR_RATIO )

        if top_dependence == shape_dependence:
            return top_dependence

        if shape_dependence:
            if top_ratio < SUFFICIENT_TOP_RATIO:
                from molass_legacy.KekLib.BasicUtils import ordinal_str
                logger.info( '%s peak has been judged as dependent with diff_ratio=%.3g' % ( ordinal_str(k+1), diff_ratio) )
                return shape_dependence

        return top_dependence

    def identify_ignorable_elements(self):
        if self.global_flag:
            return np.zeros(len(self.opt_recs))

        assert len(self.opt_recs) == len(self.opt_recs_uv)

        flags = []
        for k, p in enumerate(self.proportions):
            flag = ( p < IGNORABLE_PROPORTION
                    or self.marked_as_dependent[k] == 1)
            flags.append(flag)

        return np.array(flags)

    def compute_element_score(self, num_elements):
        count = num_elements
        for rec in self.opt_recs_uv:
            if rec.peak.sign < 0:
                count += 1
        return self.nresid_uv * count

    def compute_synthetic_score(self, ignorable_flags=None, select_matrix=None, debug=False):
        if ignorable_flags is None:
            ignorable_flags = self.identify_ignorable_elements()
        if select_matrix is None:
            info = self.get_range_edit_info()
            select_matrix = info.select_matrix

        if debug:
            print('self.nresid_uv=', self.nresid_uv)

        scores = []
        for k in range(len(self.opt_recs)):
            if ignorable_flags[k] == 0:
                ee_select = select_matrix[k]
                # self.logger.info('select_matrix[%d]=%s' % (k, str(ee_select)))
                scores.append( self.compute_element_score(np.sum(ee_select)) )
        score = np.average(scores)

        if debug:
            print('score=', score)

        return score

    def remove_unwanted_elements(self):
        import logging
        logger = logging.getLogger(__name__)

        self.proportions = compute_area_proportions(self.x, self.opt_recs)
        logger.info("debug: decomposed proportions are %s", str(self.proportions))
        to_be_removed = np.where(self.proportions < AREA_PROPORTION_LIMIT)[0]
        if len(to_be_removed) == 0:
            # nothing to remove
            return

        logger.info("decomposed proportions are %s", str(self.proportions))

        n = len(self.opt_recs)

        for k in to_be_removed[::-1]:
            del self.opt_recs[k]
            del self.opt_recs_uv[k]

        # recompute proportions
        # calling self.set_proportions() instead here results in a strange behavior. why?
        self.proportions = compute_area_proportions(self.x, self.opt_recs)

        logger.info("unwanted small-proportion %s-th elements have been removed from decomposition results.", str(to_be_removed))
        logger.info("recomputed proportions are %s", str(self.proportions))

    def set_proportions(self):
        self.proportions = compute_area_proportions(self.x, self.opt_recs)