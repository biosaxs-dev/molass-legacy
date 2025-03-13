"""
    SimpleDecompose.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Characteristic import CfSpace
from molass_legacy.Decomposer.FitRecord import FitRecord
from molass_legacy.Decomposer.ModelEvaluator import ModelEvaluator
from molass_legacy.Decomposer.UnifiedDecompResult import UnifiedDecompResult
from Selective.PeakProxy import PeakProxy
from molass_legacy.Models.CfsEvalPeaks import CFSE_WEIGHT
from Distance.JensenShannon import deformed_jsd

class NotImplementedError(Exception): pass

IGNORABLE_PROPORTION = 0.01
PENALTY_SCALE = 1000

def decompose_curve(x, arg_y, model, num_peaks, props=None, using_hybrid=False, using_cfs=False, logger=None, debug=False):
    y = arg_y.copy()
    y[y < 0] = 0        # negative values are not allowed in distance.jensenshannon()

    model_params = model.guess_multiple(x, y, num_peaks, debug=debug)
    num_params = len(model_params)      # num_params can be less than num_peaks in cases like pH6

    b = np.percentile(y, 95)

    if using_hybrid or using_cfs:
        cfs = CfSpace()
        cft0 = cfs.compute_cf(x, y)

    if props is not None:
        props = np.asarray(props)

    def objective(params, debug=False, debug_title=None):
        cy_list = []
        if props is not None:
            areas = []
        for p in params.reshape(model_params.shape):
            cy = model.func(x, *p)
            cy_list.append(cy)
            if props is not None:
                areas.append(np.sum(cy))
        ty = np.sum(cy_list, axis=0)

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(debug_title)
                ax.plot(x, y)
                for cy in cy_list:
                    ax.plot(x, cy, ":")
                ax.plot(x, ty, ":", color="red", lw=3)
                fig.tight_layout()
                plt.show()

        if using_hybrid:
            cft1 = cfs.compute_cf(x, ty)
            return np.log(np.sum((ty - y)**2)) + CFSE_WEIGHT*np.log(np.sum((cft1 - cft0)**2))
            # return np.log(np.sum((ty - y)**2)) + CFSE_WEIGHT*np.log(np.sum(np.abs(cft1 - cft0)**2))
        elif using_cfs:
            cft1 = cfs.compute_cf(x, ty)
            return np.sum((cft1 - cft0)**2)
            # return np.sum(np.abs(cft1 - cft0)**2)
        else:
            # fv = deformed_jsd(ty, y, b=b)
            ydiv = np.sum((ty - y)**2)
            if props is None:
                fv = ydiv
            else:
                props_ = np.array(areas)/np.sum(areas)
                pdiv = np.sum((props_ - props)**2)
                fv = np.log10(ydiv)*0.8 + np.log10(pdiv)*0.2
                if debug:
                    print("ydiv, pdiv=", ydiv, pdiv)

            if np.isnan(fv):
                # as in AhRR with EDM
                fv = np.inf
            return fv

    params = model_params.flatten()
    if debug:
        objective(params, debug=True, debug_title="before minimize")

    ret_edm = minimize(objective, params)
    if debug:
        objective(ret_edm.x, debug=True, debug_title="after minimize")

    opt_recs = []
    kno = 0
    area = np.sum(y)
    for k, params in enumerate(ret_edm.x.reshape(model_params.shape)):
        cy = model.func(x, *params)       # make it less expensive
        j = np.argmax(cy)
        peak = PeakProxy(top_x=x[j], top_y=cy[j])
        carea = np.sum(cy)
        area_prop = carea/area
        if area_prop > IGNORABLE_PROPORTION:
            fit_rec = FitRecord(kno, ModelEvaluator(model, params, sign=1), 0, peak, area=carea)
            opt_recs.append(fit_rec)
            kno += 1
        else:
            logger.info("element with area_prop=%g <= %g has been ingored in decompose_curve.", area_prop, IGNORABLE_PROPORTION)

    return opt_recs

def remove_the_smallest(x, opt_recs, recsname, logger):
    heights = []
    for rec in opt_recs:
        y = rec.evaluator(rec.peak.top_x)
        heights.append(y)
    i = np.argmin(heights)
    opt_recs.pop(i)

    logger.warning("removed the smallest %d-th component from %s, where heights=%s", i, recsname, str(heights))

    return opt_recs

def plot_both_results(x, y, uv_y, opt_recs, opt_recs_uv, title):

    def plot_results(ax, x, y, recs):
        ax.plot(x, y)
        cy_list = []
        for rec in recs:
            cy = rec.evaluator(x)
            ax.plot(x, cy, ":")
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        if len(ty) == len(x):
            # temporary fix for ValueError: x and y must have same first dimension, but have shapes (926,) and (1,)
            ax.plot(x, ty, ":", color="red", lw=3)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle(title)
        ax1.set_title("UV")
        ax2.set_title("XR")
        plot_results(ax1, x, uv_y, opt_recs_uv)
        plot_results(ax2, x, y, opt_recs)
        fig.tight_layout()
        ret = plt.show()

    return ret

def sort_and_renumber(opt_recs):
    opt_recs = sorted(opt_recs, key=lambda rec: rec.peak.top_x)
    for k, rec in enumerate(opt_recs):
        rec.kno = k
    return opt_recs

def decompose_elution_simply(x, y, uv_y, model, traditional_info, props=None, using_cfs=False, debug=False):

    logger = logging.getLogger(__name__)

    num_peaks = traditional_info.num_peaks
    mapper = traditional_info.mapper

    opt_recs = decompose_curve(x, y, model, num_peaks, props=props, using_cfs=using_cfs, logger=logger, debug=debug)
    opt_recs_uv = decompose_curve(x, uv_y, model, num_peaks, props=props, using_cfs=using_cfs, logger=logger, debug=debug)

    if debug:
        from molass_legacy.Decomposer.OptRecsUtils import compute_area_proportions
        uv_props = compute_area_proportions(x, opt_recs_uv)
        print("uv_props=", uv_props)
        xr_props = compute_area_proportions(x, opt_recs)
        print("xr_props=", xr_props)
        ret = plot_both_results(x, y, uv_y, opt_recs, opt_recs_uv, "decomposed")
        if not ret:
            return

    if abs(len(opt_recs) - len(opt_recs_uv)) > 0:
        if len(opt_recs) < len(opt_recs_uv):
            while len(opt_recs) < len(opt_recs_uv):
                opt_recs_uv = remove_the_smallest(x, opt_recs_uv, "opt_recs_uv", logger)
        else:
            while len(opt_recs) > len(opt_recs_uv):
                opt_recs = remove_the_smallest(x, opt_recs, "opt_recs", logger)

    if len(opt_recs) != len(opt_recs_uv):
        if debug:
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("decompose_elution_simply debug")
                ax1.plot(x, uv_y)
                for rec in opt_recs_uv:
                    cy = rec.evaluator(x)
                    ax1.plot(x, cy, ":")
                ax2.plot(x, y)
                for rec in opt_recs:
                    cy = rec.evaluator(x)
                    ax2.plot(x, cy, ":")
                fig.tight_layout()
                plt.show()

        raise NotImplementedError("len(opt_recs) %d != len(opt_recs_uv) %d" % (len(opt_recs), len(opt_recs_uv)))

    opt_recs = sort_and_renumber(opt_recs)
    opt_recs_uv = sort_and_renumber(opt_recs_uv)

    if debug:
        from molass_legacy.Decomposer.OptRecsUtils import compute_area_proportions
        uv_props = compute_area_proportions(x, opt_recs_uv)
        print("uv_props=", uv_props)
        xr_props = compute_area_proportions(x, opt_recs)
        print("xr_props=", xr_props)
        plot_both_results(x, y, uv_y, opt_recs, opt_recs_uv, "sorted")

    simple_result = UnifiedDecompResult(
                    xray_to_uv=None,
                    x_curve=mapper.x_curve, x=x, y=y,
                    opt_recs=opt_recs,
                    max_y_xray = np.max(y),     # get this from traditional_info
                    model_name=model.get_name(),
                    decomposer=None,            # what is this used for?
                    uv_y=uv_y,
                    opt_recs_uv=opt_recs_uv,
                    max_y_uv = np.max(uv_y),    # get this from traditional_info
                    )

    simple_result.set_area_proportions()
    simple_result.remove_unwanted_elements()

    return simple_result
