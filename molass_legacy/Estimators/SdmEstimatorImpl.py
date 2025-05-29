"""
    Estimators.SdmEstimatorImpl.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Models.Stochastic.RgReliability import determine_unreliables

def guess_exec_spec(peak_rgs, props, qualities):
    """
    for these settings, see also SecTheory.ColumnTypes.py, Optimizer.ptimizerSettings.py
    """
    unreliable_indeces = determine_unreliables(peak_rgs, qualities, props, trust_max_num=None)
    poresize_bounds = get_setting("poresize_bounds")
    init_N0 = get_setting("num_plates_pc")
    return {"unreliable_indeces": unreliable_indeces, 'poresize_bounds': poresize_bounds, 'init_N0': init_N0}

def edit_to_full_sdmparams(editor, sdm_params, corrected_rgs, uv_curve, debug=False):
    if debug:
        from importlib import reload
        import molass_legacy.Models.Stochastic.MonoporeUvScaler
        reload(molass_legacy.Models.Stochastic.MonoporeUvScaler)
        
    from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCUV_PARAMS
    from molass_legacy.Models.Stochastic.DispersiveUvScaler import adjust_to_uv_scales

    # (xr_curve, D), rg_curve, (uv_curve, U) = editor.dsets

    N, K, x0, poresize, N0, tI = sdm_params[0:NUM_SDMCUV_PARAMS]
    T = K/N
    me = 1.5
    mp = 1.5
    xr_w = sdm_params[NUM_SDMCUV_PARAMS:]
    rgs = corrected_rgs
    sdmcol_params = np.array([N, K, x0, poresize, N0, tI])      # N, K, x0, poresize, N0, tI

    _, _, xr_x, xr_y, baselines = editor.get_curve_xy(return_baselines=True)
    a, b = editor.peak_params_set[-2:]
    uv_x = xr_x*a + b
    uv_y = uv_curve.spline(uv_x)

    uv_baseline = editor.get_uv_baseline_deprecated(xy=(uv_x, uv_y))    # 

    uv_y_ = uv_y - uv_baseline
    xr_y_ = xr_y - baselines[1]

    ret = adjust_to_uv_scales(xr_x, xr_y_, uv_x, uv_y_, sdm_params, corrected_rgs, debug=debug)
    if ret is None:
        return
    else:
        uv_w, uv_ty = ret

    uv_base_params = editor.get_uv_base_params(xyt=(uv_x, uv_y, uv_ty))
    param_list = [xr_w, editor.baseline_params[1], rgs, (a, b), uv_w, uv_base_params, xr_x[[0,-1]], sdmcol_params]
    init_params = np.concatenate(param_list)

    if debug:
        # self.logger.info("uv_base_params=%s, average(uv_ty)=%g", str(uv_base_params), np.average(uv_ty))
        uv_bl_computed = editor.base_curve_info[0](uv_x, uv_base_params, uv_ty)
        with plt.Dp():
            fig,(ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("compute_sdm_init_params")
            ax1.plot(uv_x, uv_y)
            ax1.plot(uv_x, uv_ty + uv_baseline, ":")
            ax1.plot(uv_x, uv_baseline, color="red")
            ax1.plot(uv_x, uv_bl_computed, color="yellow")
            ax2.plot(xr_x, xr_y)
            ax2.plot(xr_x, baselines[1], color="red")
            fig.tight_layout()
            plt.show()

    return init_params
