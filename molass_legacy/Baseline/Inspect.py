"""
    Basseline.Inspect.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Batch.LiteBatch import LiteBatch

def baseline_inspect_impl(caller):
    print('baseline_inspect_impl')
    
    lb = LiteBatch()
    lb.prepare(caller.serial_data, debug=True)
    uv_x, uv_y, xr_x, xr_y, baselines = lb.get_curve_xy(return_baselines=True, debug=False)
    uv_y_ = uv_y - baselines[0]
    xr_y_ = xr_y - baselines[1]
    # uv_peaks, xr_peaks = lb.get_modeled_peaks(uv_x, uv_y_, xr_x, xr_y_)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        fig.suptitle("Baseline inspection Result")
        ax1.set_title("UV")
        ax1.plot(uv_x, uv_y)
        ax1.plot(uv_x, baselines[0], color="red")

        ax2.set_title("XR")
        ax2.plot(xr_x, xr_y)
        ax2.plot(xr_x, baselines[1], color="red")
        
        fig.tight_layout()
        plt.show()