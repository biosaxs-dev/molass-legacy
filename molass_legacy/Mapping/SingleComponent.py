# coding: utf-8
"""

    SingleComponent.py

    Copyright (c) 2018-2021, SAXS Team, KEK-PF

"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

PEAK_EVAL_RANGE_RATIO   = 0.05
DIFF_ALLOW_LIMIT_RATIO  = 0.02
USE_SMOOTHED_XRAY_CURVE = True
ACCEPTABLE_MIN_SCI  = 80
SCI_BOUNDARIES = [50, ACCEPTABLE_MIN_SCI]

if USE_SMOOTHED_XRAY_CURVE:
    from molass_legacy.Elution.CurveUtils import smoothed_curve_y

class SingleComponent:
    """
        created just after every mapping optimization
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def compute_sci_ratio( self, debug=False ):
        mapper = self.mapper

        x = mapper.x_curve.x
        # y = mapper.x_curve.y
        y = mapper.x_curve_y_adjusted
        peak_top_y = y[mapper.x_curve.primary_peak_i]

        if USE_SMOOTHED_XRAY_CURVE:
            sy = smoothed_curve_y(mapper.x_curve, y=y)
            std_diff = np.std(y - sy)
            std_diff_ratio = std_diff/peak_top_y
        else:
            sy = y
            std_diff_ratio = 0

        allow_ratio = DIFF_ALLOW_LIMIT_RATIO + std_diff_ratio

        if debug:
            from matplotlib.patches import Rectangle
            from molass_legacy._MOLASS.SerialSettings import get_setting
            fig = plt.figure(figsize=(16,6))
            ax = fig.gca()
            ax_ = ax.twinx()
            in_folder = get_setting('in_folder')
            using = ' using smoothed xray elution' if USE_SMOOTHED_XRAY_CURVE else ''
            ax.set_title("Calulation of Single Component Indicator for " + in_folder + using, fontsize=16)

            sum_diff = 0
            set_label = True
            ax.plot( x, y, color='orange' )
            # ax.plot( x, mapper.x_curve.y, ':', color='orange' )
            if USE_SMOOTHED_XRAY_CURVE:
                ax.plot( x, sy, ':', color='red', label='smoothed xray' )
            ax.plot( x, mapper.mapped_vector, ':', color='blue', label='mapped uv' )
            xmin, xmax = x[[0, -1]]
            ax_.set_xlim(xmin, xmax)

            if USE_SMOOTHED_XRAY_CURVE:
                r = allow_ratio
                ax_.plot([xmin, xmax], [allow_ratio, allow_ratio], color='pink', label='adjusted allowance level')
                ax_.bar(x, abs(y-sy)/peak_top_y, color='gray', label='noise=abs(y - sy)/peak_top_y', alpha=0.2)
            r = DIFF_ALLOW_LIMIT_RATIO
            ax_.plot([xmin, xmax], [r, r], color='red', label='fixed allowance level')

            print('std_diff_ratio=', std_diff_ratio)
            print('mapper.peak_eval_ranges=', mapper.peak_eval_ranges)

        ratio_list = []
        for lower, mid, upper in mapper.peak_eval_ranges:
            slice_ = slice(lower, upper+1)
            y_  = sy[slice_]
            mv_ = mapper.mapped_vector[slice_]
            diff = np.abs( mv_ - y_ )

            ratio = diff / peak_top_y
            rwh = np.where( ratio > allow_ratio )[0]

            if debug:
                sum_diff += np.sum( diff )
                x_  = x[slice_]
                ax_.bar( x_, ratio, color='green', label='diff=(my - sy)/peak_top_y' if set_label else None, alpha=0.2)
                ax.plot( x_[rwh], mv_[rwh], 'o', color='yellow', label='badly matching points' if set_label else None )
                f, t = mapper.x_x[[lower, upper]]
                ymin, ymax = ax.get_ylim()
                p = Rectangle(
                        (f, ymin),  # (x,y)
                        t - f,   # width
                        ymax - ymin,    # height
                        facecolor   = 'cyan',
                        alpha       = 0.1,
                    )
                ax.add_patch(p)
                set_label = False

            if len(ratio) > 0:
                ratio = (len(y_) - len(rwh)) / len(ratio)
            else:
                ratio = np.nan
            ratio_list.append(ratio)

        if debug:
            print( 'sum_diff=', sum_diff )
            ax.legend(loc='upper left', fontsize=16)
            ax_.legend(loc='upper right', fontsize=16)
            plt.tight_layout()
            plt.show()

        # print( 'ratio_list=', ratio_list )
        assert len(ratio_list) == len(mapper.peak_eval_ranges)
        return ratio_list

    def compute_sci( self, debug=False ):
        sci_list = [ r*100 for r in self.compute_sci_ratio(debug=debug) ]
        return np.array(sci_list)
