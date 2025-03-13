"""
    Models/Stochastic/PsdEstimate.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import seaborn as sns
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.EGH import egh
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.Models.ModelUtils import plot_component_curves, get_paired_ranges_from_params_array
from molass_legacy.KekLib.ContextUtils import get_variables_from_context
from molass_legacy.Models.EGH import EGH, EGHA

sns.set_theme()

def psd_estimate_by_paired_ranges(folder, batch, context, affine=True, with_cf=False, separate_ratio=None,  debug=False, count=0,
                                  computed_params=None, compute_residual_ratio=False, use_lrf_src=False):
    logger = logging.getLogger(__name__)
    logger.info("estimating with psd_estimate_by_paired_ranges")

    uv_x, uv_y, uv_peaks = get_variables_from_context(context, 'uv_x, uv_y, uv_peaks')
    baselines, = get_variables_from_context(context, 'baselines')   # note the results are returned as a list
    xr_x, xr_y, xr_peaks = get_variables_from_context(context, 'xr_x, xr_y, xr_peaks')

    if False:
        with plt.Dp(button_spec=["OK", "Cancel"]):
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
            fig.suptitle("Impl Plot for %s" % get_in_folder(folder))
            plot_component_curves(ax1, uv_x, uv_y, uv_peaks, baselines[0], color='blue', affine=affine)
            plot_component_curves(ax2, xr_x, xr_y, xr_peaks, baselines[1], color='orange', affine=affine)
            fig.tight_layout()
            ret = plt.show()
        if not ret:
            return

    sd, = get_variables_from_context(context, 'sd')

    from Selective.LrfSource import LrfSource
    lrf_src_args1 = [uv_x, uv_y, xr_x, xr_y, baselines]
    peak_params_set = batch.get_peak_params_set()
    lrf_src = LrfSource(sd, batch.corrected_sd, lrf_src_args1, peak_params_set)
    return pds_estimate_impl(folder, lrf_src, compute_residual_ratio=compute_residual_ratio)

def pds_estimate_impl(folder, lrf_src, count=0, compute_residual_ratio=False, debug=False):
    from importlib import reload
    import Models.Stochastic.MomentsStudy
    reload(Models.Stochastic.MomentsStudy)
    from molass_legacy.Models.Stochastic.MomentsStudy import moments_study_impl
    from molass_legacy.Models.Stochastic.LognormalPoreColumn import compute_residual_ratio_impl
    from molass_legacy.Models.Stochastic.LnporeUtils import plot_lognormal_fitting_state

    lnp_params, rgs = moments_study_impl(lrf_src, return_rgs=True, debug=debug)
    x = lrf_src.xr_x
    y = lrf_src.xr_y
    residual = compute_residual_ratio_impl(x, y, rgs, lnp_params, debug=False)
    if compute_residual_ratio:
        return residual

    print("[lnp_params, [residual]]=", [lnp_params, [residual]])
    pore_params = np.concatenate([lnp_params, [residual]])
    title = "PSD Estimate Result for " + get_in_folder(folder)
    plot_lognormal_fitting_state(x, y, lnp_params, rgs, plot_boundary=True, title=title, save_fig_as='temp/fig-%03d.png' % count)
    return pore_params