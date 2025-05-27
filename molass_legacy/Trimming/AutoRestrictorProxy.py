"""
    Trimming.AutoRestrictorProxy.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
from  molass_legacy.Trimming.AutoRestrictor import get_exact_emg_peaks_list, SIGMA_POINT_RATIO

class AutoRestrictorProxy:
    def __init__(self, sd, uv_restrict_list, xr_restrict_list):
        # self.result_info = sd, curves, emg_peaks_list, old_e_restricts, ret_info, SIGMA_POINT_RATIO, self.mpeaks
        # minimum info for trimming_result_plot_impl
        uv_curve = sd.get_uv_curve()
        xr_curve = sd.get_xr_curve()
        curves = uv_curve, xr_curve

        uv_peaks, xr_peaks = get_exact_emg_peaks_list(uv_curve, xr_curve, debug=False)
        emg_peaks_list = uv_peaks, xr_peaks

        old_e_restricts = None,None
        ret_info = [uv_restrict_list, xr_restrict_list]
        mpeaks = None
        self.result_info = sd, curves, emg_peaks_list, old_e_restricts, ret_info, SIGMA_POINT_RATIO, mpeaks