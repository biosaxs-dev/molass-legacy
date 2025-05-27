"""
    DefaultNumPeaks.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Optimizer.OptConstants import MAX_NUM_PEAKS

def get_default_num_peaks(sd):
    try:
        xr_num_peaks = len(sd.get_xray_curve().peak_info)
        uv_num_peaks = len(sd.get_uv_curve().peak_info)
        if xr_num_peaks == uv_num_peaks:
            num_peaks = xr_num_peaks + 2
        else:
            num_peaks = max(xr_num_peaks, uv_num_peaks) + 1
        ret_num_peaks = min(MAX_NUM_PEAKS, num_peaks)
    except:
        # AttributeError: 'NoneType' object has no attribute 'get_xray_curve'
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, "get_default_num_peaks failed: ", n=5)
        raise RuntimeError

    return ret_num_peaks
