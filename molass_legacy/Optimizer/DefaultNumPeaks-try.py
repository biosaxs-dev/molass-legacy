"""
    DefaultNumPeaks.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Optimizer.OptConstants import MAX_NUM_PEAKS

def get_default_num_peaks(sd):
    xr_curve = sd.get_xray_curve()
    uv_curve = sd.get_uv_curve()
    try:
        xr_num_peaks = len(xr_curve.peak_info)
        uv_num_peaks = len(uv_curve.peak_info)
        if xr_num_peaks == uv_num_peaks:
            num_peaks = xr_num_peaks + 2
            plus = 1
            look_curve = xr_curve
        else:
            num_peaks = max(xr_num_peaks, uv_num_peaks) + 1
            plus = 0
            if xr_num_peaks > uv_num_peaks:
                look_curve = xr_curve
            else:
                look_curve = uv_curve
        ret_num_peaks = min(MAX_NUM_PEAKS, num_peaks)
    except:
        # AttributeError: 'NoneType' object has no attribute 'get_xray_curve'
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, "get_default_num_peaks failed: ", n=5)
        raise RuntimeError

    hs = []
    for epeak in look_curve.get_emg_peaks():
        h = epeak.get_params()[0]
        hs.append(h)
    i = np.argmax(hs)
    if plus > 0:
        n = i + plus
    else:
        n = None
    return ret_num_peaks, n
