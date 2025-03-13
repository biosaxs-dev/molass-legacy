"""
    Simulative.BridgeUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Decomposer.FitRecord import FitRecord
from molass_legacy.Decomposer.ModelEvaluator import ModelEvaluator
from Selective.PeakProxy import PeakProxy
from molass_legacy.QuickAnalysis.ModeledPeaks import get_modeled_peaks_impl
from molass_legacy.Decomposer.UnifiedDecompResult import UnifiedDecompResult

IGNORABLE_PROPORTION = 0.01

def make_traditional_fitrecs(x, y, model, peaks, debug=False):
    opt_recs = []
    kno = 0
    area = np.sum(y)
    for k, params in enumerate(peaks):
        cy = model(x, params)
        j = np.argmax(cy)
        peak = PeakProxy(top_x=x[j], top_y=cy[j])
        carea = np.sum(cy)
        area_prop = carea/area
        if area_prop > IGNORABLE_PROPORTION:
            fit_rec = FitRecord(kno, ModelEvaluator(model, params, sign=1), 0, peak, area=carea)
            opt_recs.append(fit_rec)
            kno += 1
        else:
            logger = logging.getLogger(__name__)
            logger.info("element with area_prop=%g <= %g has been ingored in decompose_curve.", area_prop, IGNORABLE_PROPORTION)

    return opt_recs

def decompose_by_bridge(x, y, uv_y, model, traditional_info, debug=False):
    from Selective.BridgeUtils import make_traditional_fitrecs
    num_peaks = traditional_info.num_peaks
    mapper = traditional_info.mapper
    print("num_peaks=", num_peaks)

    if debug:
        def plot_curves(ax, x, y, recs=None):   
            ax.plot(x, y, label="data")
            if recs is not None:
                cy_list = []
                for k, rec in enumerate(recs):
                    cy = rec.evaluator(x)
                    ax.plot(x, cy, ":", label="component-%d" % k)
                    cy_list.append(cy)
                ty = np.sum(cy_list, axis=0)
                ax.plot(x, ty, ":", color="red", label="model total")
            ax.legend()

        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("decompose_by_bridge input")
            plot_curves(ax1, x, uv_y)
            plot_curves(ax2, x, y)
            fig.tight_layout()
            plt.show()

    # task: make get_modeled_peaks_impl support other models than EGHA
    uv_peaks, xr_peaks = get_modeled_peaks_impl(1, 0, x, uv_y, x, y, num_peaks, exact_num_peaks=num_peaks, affine=True)
    opt_recs = make_traditional_fitrecs(x, y, model, xr_peaks)
    opt_recs_uv = make_traditional_fitrecs(x, uv_y, model, uv_peaks)

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("decompose_by_bridge results")
            plot_curves(ax1, x, uv_y, recs=opt_recs_uv)
            plot_curves(ax2, x, y, recs=opt_recs)
            fig.tight_layout()
            plt.show()

    result = UnifiedDecompResult(
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

    result.set_area_proportions()
    result.set_proportions()    # do not remove_unwanted_elements, set proportions only.

    return result