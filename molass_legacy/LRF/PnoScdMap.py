"""
    PnoScdMap.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting

class PnoScdMap:
    def __init__(self, sd, cnv_ranges, debug=False):
        """
        using the paired peak_top_x here because
        self.sd.xray_curve.peak_top_x may have been changed
        from this peak_top_x as in 20200121_2

        also note that these coordinates are in relative values,
        i.e., x - x[0]
        """
        self.logger = logging.getLogger(__name__)
        peak_top_x, scd_colors = get_setting('mapper_cd_color_info')
        pno_map = []
        ecurve = sd.get_xray_curve()
        main_y = None
        for pno, prange in enumerate(cnv_ranges):
            x = prange.top_x
            dist = np.abs(peak_top_x - x)
            k = np.argmin(dist)
            if dist[k] < 5:
                # i.e., this pno-th prange corresponds to the k-th point in peak_top_x
                pass
            else:
                # i.e., there is no corresponding point in peak_top_x with this prange
                k = None
            pno_map.append(k)

            y = ecurve.spline(x)
            if main_y is None or y > main_y:
                main_y = y
                self.main_pno = pno

        self.pno_map = pno_map
        self.scd_colors = scd_colors
        self.logger.info("pno_map=%s, scd_colors=%s", str(pno_map), str(scd_colors))
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            ecurve = sd.xray_curve
            x = ecurve.x
            y = ecurve.y
            print("cnv_ranges=", cnv_ranges)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.plot(x, y)
                for prange in cnv_ranges:
                    pass
                fig.tight_layout()
                plt.show()

    def get_color(self, pno):
        i = self.pno_map[pno]
        if i is None:
            return '???'
        else:
            return self.scd_colors[i]

    def get_main_peak_rank(self):
        color = self.get_color(self.main_pno)
        return 2 if color in ["red", "yellow", "orange"] else 1
