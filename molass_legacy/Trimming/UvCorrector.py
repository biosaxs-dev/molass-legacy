"""
    UvCorrector.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import linregress
from molass_legacy.KekLib.ScipyUtils import get_spline

NUM_KNOTS = 8
FIT_TAIL_RATIO = 0.005

class UvCorrector:
    def __init__(self, curve1, curve2):
        self.curve1 = curve1
        self.curve2 = curve2
        self.x = curve2.x
        self.y = curve2.y

    def correction_curve(self, x, *params):
        assert False

    def fit(self, peak_region, debug=False, fig_file=None):
        assert False

    def get_widest_possible_fit_slice(self, peak_region, debug=False):
        x = self.x
        y = self.y
        proportions = [FIT_TAIL_RATIO, 1 - FIT_TAIL_RATIO]
        _, (f, t) = peak_region.get_peak_ends(proportions)

        inside = np.zeros(len(x), dtype=bool)
        inside[f:f+5] = True
        inside[t-4:t+1] = True

        fit_slice = slice(f, t+1)
        outside = np.ones(len(x), dtype=bool)
        outside[fit_slice] = False
        params = []
        for i in [inside, outside]:
            slope, intercept = linregress(x[i], y[i])[0:2]
            params.append((slope, intercept))
        params = np.array(params)
        slope0, slope1 = params[:,0]
        slope_deviation = abs((slope1/slope0 - 1)*len(x)/(y[-1] - y[0]))
        print("slope_deviation=", slope_deviation)
        if True:
            # ret_f, ret_t = 0, len(x)-1
            ret_f, ret_t = f, t
        else:
            ret_f, ret_t = self.find_widest_possible_ends(x, y, f, t, params)

        ret_slice = slice(ret_f, ret_t+1)
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_widest_possible_fit_slice")
                ax.plot(x, y)
                for slope, intercept in params:
                    ax.plot(x, x*slope + intercept)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                for i, p in enumerate([f, t, ret_f, ret_t]):
                    color = "gray" if i < 2 else "cyan"
                    ax.plot([p, p], [ymin, ymax], ":", color=color)
                fig.tight_layout()
                plt.show()

        return ret_slice

    def find_widest_possible_ends(self, x, y, f, t, params):
        return f, t

    def get_corrected_y(self, params=None):
        if params is None:
            params = self.popt
        slope, intercept = params[0:2]
        params_ = params.copy()
        params_[0:2] = 0
        dy_ = self.correction_curve(self.x, *params_)
        return self.y - dy_

    def get_corrected_ppn(self, sy, k, spp3_extra, debug=False):
        # may be deprecated
        x = self.x
        spline = get_spline(x, sy, num_knots=NUM_KNOTS)
        d_spline = spline.derivative(1)
        sspp = sorted(spp3_extra, key=lambda j: -abs(d_spline(j)))

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            csy = spline(x)

            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_corrected_ppn")
                ax.plot(x, sy)
                ax.plot(x, csy, color="cyan")
                for j in sspp:
                    ax.plot(x[j], csy[j], "o", label=str(j))
                ax.legend()
                axt = ax.twinx()
                axt.plot(x, d_spline(x), color="yellow")
                fig.tight_layout()
                plt.show()

        return sspp[0:k]
