"""
    QuickAnalysis.PreDecomposer.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Batch.FullBatch import FullBatch
from molass_legacy.UV.UvPreRecog import UvPreRecog

USE_LRF = False
if USE_LRF:
    from .ModeledPeaks import plot_curve, get_proportions, PeakRegionLrf
else:
    from .ModeledPeaks import plot_curve, get_proportions, get_peak_region_ends

class PreDecomposer(FullBatch):
    def __init__(self, sd, pre_recog, sd_copy, debug=False):
        print("PreDecomposer.__init__")
        self.logger = logging.getLogger(__name__)
        self.sd = sd
        self.pre_recog = pre_recog
        self.sd_copy = sd_copy
        self.xr_curve = sd.get_xray_curve()
        self.ecurve_info = None
        self.unified_baseline_type = get_setting("unified_baseline_type")
        self.upr = UvPreRecog(sd, pre_recog, debug=debug)
        self.base_curve_info = self.upr.get_base_curve_info()

    def update_sd_copy(self, sd_copy):
        self.sd_copy = sd_copy

    def decompose(self, debug=False):
        uv_x, uv_y, xr_x, xr_y, baselines = self.get_curve_xy(return_baselines=True)
        uv_y_ = uv_y - baselines[0]
        xr_y_ = xr_y - baselines[1]
        num_peaks = len(self.xr_curve.peak_info) + 2
        uv_peaks, xr_peaks = self.get_modeled_peaks(uv_x, uv_y_, xr_x, xr_y_, num_peaks=num_peaks)

        if USE_LRF:
            D, E, qv, _ = self.sd.get_xr_data_separate_ly()
            lrf = PeakRegionLrf(D, E, xr_x, xr_peaks)
            f, t = lrf.ends
            rgs = lrf.compute_rgs(qv)
            print("rgs=", rgs)
        else:
            f, t = get_peak_region_ends(xr_x, xr_peaks)

        if debug:
            from matplotlib.patches import Rectangle
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("decompose: debug", fontsize=20)

                ax1.set_title("UV Elution", fontsize=16)
                ax2.set_title("XR Elution", fontsize=16)

                plot_curve(ax1, uv_x, uv_y, uv_peaks, color='blue', baseline=baselines[0])
                plot_curve(ax2, xr_x, xr_y, xr_peaks, color='orange', baseline=baselines[1])

                ymin, ymax = ax2.get_ylim()
                p = Rectangle(
                        (f, ymin),      # (x,y)
                        t - f,          # width
                        ymax - ymin,    # height
                        facecolor   = 'cyan',
                        alpha       = 0.2,
                    )
                ax2.add_patch(p)

                fig.tight_layout()
                plt.show()

        self.uv_x = uv_x
        self.xr_x = xr_x
        self.uv_peaks = uv_peaks
        self.xr_peaks = xr_peaks

    def get_xr_proportions(self):
        return get_proportions(self.xr_x, self.xr_peaks)

def spike(caller):
    from time import time
    from importlib import reload
    import QuickAnalysis.Homopeak
    reload(QuickAnalysis.Homopeak)
    from .Homopeak import Homopeakness

    t0 = time()
    sd = caller.sd_orig
    sd_copy = caller.serial_data
    pre_recog = caller.pre_recog_orig
    predec = PreDecomposer(sd, pre_recog, sd_copy, debug=False)
    predec.decompose(debug=True)
    hpn = Homopeakness(predec)
    hpn.get_homopeakness_scores()
    print("it took ", time() - t0)
