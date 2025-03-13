"""
    Peaks.PeakEditorUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from molass_legacy.QuickAnalysis.ModeledPeaks import get_modeled_peaks_impl
from Selective.LrfSource import LrfSource

def apply_new_peaks(editor, new_peaks, debug=False):
    print("apply_new_peaks")

    a, b = editor.peak_params_set[2:4]
    uv_x, uv_y, xr_x, xr_y = editor.lrf_src_args1[0:4]
    num_peaks = len(new_peaks)

    # replace peak recognition with peaks=new_peaks
    uv_peaks, xr_peaks = get_modeled_peaks_impl(a, b, uv_x, uv_y, xr_x, xr_y, num_peaks, peaks=new_peaks, debug=debug)
    editor.peak_params_set.xr_peaks = xr_peaks
    editor.peak_params_set.uv_peaks = uv_peaks
    lrf_src = LrfSource(editor.sd, editor.corrected_sd, editor.lrf_src_args1, editor.peak_params_set)

    estimator = editor.fullopt.params_type.get_estimator(editor)
 
    init_params = estimator.estimate_params(lrf_src=lrf_src, debug=False)
    editor.draw_scores(init_params=init_params, create_new_optimizer=False)