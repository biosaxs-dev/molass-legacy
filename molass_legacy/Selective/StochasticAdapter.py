"""
    Selective.StochasticAdapter.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Batch.DataBridgeUtils import get_databridge
from molass_legacy.Decomposer.FitRecord import FitRecord
from molass_legacy.Decomposer.ModelEvaluator import ModelEvaluator
from molass_legacy.Models.Stochastic.MonoporeUvScaler import adjust_to_uv_scales
from molass_legacy.Selective.PeakProxy import PeakProxy

def convert_to_stochastic_decomposition(decomp_editor, debug=False):
    from importlib import reload
    import molass_legacy.Models.Stochastic.Monopore
    reload(molass_legacy.Models.Stochastic.Monopore)
    from molass_legacy.Models.Stochastic.Monopore import Monopore
    import molass_legacy.Decomposer.UnifiedDecompResult
    reload(molass_legacy.Decomposer.UnifiedDecompResult)
    from molass_legacy.Decomposer.UnifiedDecompResult import UnifiedDecompResult

    print("convert_to_stochastic_decomposition")
    
    bridge = get_databridge(decomp_editor)
    lrf_src = bridge.get_lrf_source()
    xr_x = lrf_src.xr_x
    xr_y = lrf_src.xr_y
    uv_x = lrf_src.uv_x
    uv_y = lrf_src.uv_y
    x_curve = decomp_editor.mapper.x_curve  # note that this is baseline-corrected
    a_curve = decomp_editor.mapper.a_curve  # same as above
    model = Monopore(delayed=True)          # get this from the editor
    model.set_fx_for_height_ratio(xr_x)

    mnp_params, corrected_rgs = lrf_src.guess_monopore_params(debug=debug)

    # convert mnp_params to decomp_result
    # task: unify the bridging procedure below with that in Estimator.StcEstimator.py

    a, b = bridge.peak_params_set[-2:]
    uv_x_ = a*xr_x + b                          # measured x
    uv_y_ = a_curve.spline(a*x_curve.x + b)     # note that uv_y_ is consistent with uv_x and uv_y

    if debug:
        from molass_legacy.Elution.CurveUtils import simple_plot
        print("len(xr_x)=", len(xr_x), "len(uv_x)=", len(uv_x))
        print("len(x_curve.x)=", len(x_curve.x), "len(a_curve.x)=", len(a_curve.x))
        with plt.Dp():
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
            fig.suptitle("convert_to_stochastic_decomposition inputs")
            simple_plot(axes[0,0], a_curve)
            simple_plot(axes[0,1], x_curve)
            axes[1,0].plot(uv_x, uv_y)
            axes[1,0].plot(uv_x_, uv_y_, ":")
            axes[1,1].plot(xr_x, xr_y)
            fig.tight_layout()
            ret = plt.show()
        if not ret:
            return

    xr_scales = mnp_params[6:]
    uv_scales, uv_ty = adjust_to_uv_scales(xr_x, xr_y, uv_x_, uv_y_, mnp_params, corrected_rgs, debug=debug)

    opt_recs = []
    opt_recs_uv = []

    # xr_area = np.sum(y)
    # uv_area = np.sum(uv_y)

    for kno, (xr_scale, uv_scale, rg) in enumerate(zip(xr_scales, uv_scales, corrected_rgs)):
        xr_params = np.concatenate([[xr_scale], mnp_params[:6], [rg]])
        cy = model.func(xr_x, *xr_params)
        j = np.argmax(cy)
        peak = PeakProxy(top_x=xr_x[j], top_y=cy[j])
        opt_rec = FitRecord(kno, ModelEvaluator(model, xr_params, sign=1, accepts_real_x=True), 0, peak)
        opt_recs.append(opt_rec)
        uv_params = np.concatenate([[uv_scale], mnp_params[:6], [rg]])
        cy = model.func(uv_x_, *uv_params)
        j = np.argmax(cy)
        peak = PeakProxy(top_x=uv_x_[j], top_y=cy[j])       # this is doubtfull. currently not used?
        opt_rec_uv = FitRecord(kno, ModelEvaluator(model, uv_params, sign=1, accepts_real_x=True), 0, peak)
        opt_recs_uv.append(opt_rec_uv)

    decomp_result = UnifiedDecompResult(
        xray_to_uv=None,
        x_curve=x_curve, x=xr_x, y=xr_y,
        opt_recs=opt_recs,
        max_y_xray=x_curve.max_y,
        model_name=model.get_name(),
        uv_y=uv_y_,
        opt_recs_uv=opt_recs_uv,
        max_y_uv=a_curve.max_y,
        )
    
    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("convert_to_stochastic_decomposition outputs")
            ax1.set_title("UV")
            ax1.plot(uv_x, uv_y, label="data")
            cy_list = []
            for rec in opt_recs_uv:
                cy = rec.evaluator(xr_x)
                cy_list.append(cy)
                ax1.plot(uv_x_, cy, ":", label="component-%d" % rec.kno)
            ty = np.sum(cy_list, axis=0)
            ax1.plot(uv_x_, ty, ":", color="red", lw=2, label="model total")
            ax1.legend()

            ax2.set_title("XR")
            ax2.plot(xr_x, xr_y, label="data")
            cy_list = []
            for rec in opt_recs:
                cy = rec.evaluator(xr_x)
                cy_list.append(cy)
                ax2.plot(xr_x, cy, ":", label="component-%d" % rec.kno)
            ty = np.sum(cy_list, axis=0)
            ax2.plot(xr_x, ty, ":", color="red", lw=2, label="model total")
            ax2.legend()

            fig.tight_layout()
            plt.show()

    decomp_result.set_area_proportions()
    decomp_result.remove_unwanted_elements()    # or decomp_result.set_proportions() if you should keep minor components as well

    if debug:
        decomp_result.get_range_edit_info(debug=debug)

    return decomp_result