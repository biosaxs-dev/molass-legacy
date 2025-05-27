"""

    ScatteringBaseline.py

    scattering baseline solver

    Copyright (c) 2017-2025, SAXS Team, KEK-PF

"""
import numpy as np
from scipy import stats
from molass_legacy._MOLASS.SerialSettings import get_setting

PERCENTILE_FIRST        = 25
PERCENTILE_SECOND       = 25
PERCENTILE_FINAL        = 10
VERY_SMALL_SLOPE_RATIO  = 0.01
CONVERGENCE_RATIO       = 0.1
ALTERNATING_LIMIT_RATIO = 0.5

class ScatteringBaseline:
    def __init__( self, y, x=None, height=None, curve=None, logger=None, suppress_warning=False):
        if x is None:
            if logger is None:
                import logging
                logger = logging.getLogger(__name__)

            if curve is None:
                if not suppress_warning:
                    from molass_legacy.KekLib.DebugUtils import show_call_stack
                    show_call_stack("ScatteringBaseline")
                    logger.warning("*************** using ScatteringBaseline without x is deprecated.")
                x = np.arange(len(y))
            else:
                x = curve.x
                # logger.info("x has been set from the given curve")
        self.y = y
        self.x = x

        if height is None:
            height  = np.max( y ) - np.min( y )

        self.very_small_sploe = height / len(y) * VERY_SMALL_SLOPE_RATIO
        self.curve = curve
        self.logger = logger

    def solve( self, p_final=PERCENTILE_FINAL, max_iter_num=10, no_shift=False, debug=False ):
        xpp, ypp, slope, intercept, ppp, adopted_sice = self.get_low_percentile_params( self.x, self.y, max_iter_num, debug=debug )
        y_  = self.y - slope * self.x
        if adopted_sice is not None:
            y_  = y_[adopted_sice]
        p   = np.percentile( y_, p_final )
        p_  = np.where( y_ <= p )[0]
        m   = np.argmax( y_[p_] )
        n   = p_[m]

        A   = slope
        if no_shift:
            B   = intercept
        else:
            B   = self.y[n] - slope*self.x[n]
        self.xpp    = xpp
        self.ypp    = ypp
        self.npp    = n
        self.params = ( A, B )

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            x = self.x
            y = self.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("ScatteringBaseline.solve")
                ax.plot(x, y)
                ax.plot(x, y_)
                ax.plot(x, A*x + B, color="red")
                fig.tight_layout()
                plt.show()

        return A, B

    def is_alternating( self, slope_list ):
        slope_array = np.array(slope_list)
        half_iter_num = len(slope_list)
        average_slope = np.average(slope_array)
        slope_devs = slope_array - average_slope
        total_stdev = np.std(slope_array)
        upper_ratio = np.std(slope_array[slope_devs > 0])/total_stdev
        lower_ratio = np.std(slope_array[slope_devs < 0])/total_stdev
        # print('ratios=', upper_ratio, lower_ratio)
        alternating = (upper_ratio + lower_ratio) < ALTERNATING_LIMIT_RATIO
        # this is True for 20181203 (only as of Jan. 2019)
        return alternating, average_slope

    def debug_plot(self, x, y, y_, pc, title, ab=None):
        import molass_legacy.KekLib.DebugPlot as plt
        pp = y_ <= pc
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title(title)
            ax.plot(x, y)
            ax.plot(x, y_, ":")
            ax.plot(x[pp], y[pp], "o", color="yellow")
            if ab is not None:
                a, b = ab
                ax.plot(x, a*x + b, color="red")
            fig.tight_layout()
            plt.show()

    def get_low_percentile_params( self, x, y, max_iter_num, debug=False ):
        y_  = y
        ppp = np.percentile( y_, [ PERCENTILE_FIRST ] )
        last_slope  = None
        init_diff   = None
        alternating = False
        half_iter_num = max_iter_num//2
        slope_list = []

        if debug:
            self.debug_plot(x, y, y_, ppp[0], "get_low_percentile_params: entry")

        for i in range(max_iter_num):

            if i == half_iter_num:
                alternating, average_slope = self.is_alternating( slope_list )
                if alternating:
                    if self.logger is not None:
                        self.logger.warning( 'alternating state detected in LPM' )

            if alternating:
                if i > half_iter_num:
                    average_slope = np.average(slope_list[-half_iter_num:])
                slope = average_slope
            else:
                pp = y_ <= ppp[0]
                xpp = x[pp]
                ypp = y[pp]
                slope, intercept = stats.linregress( xpp, ypp )[0:2]

                if debug:
                    self.debug_plot(x, y, y_, ppp[0], "get_low_percentile_params: [%d] result" % i, ab=(slope, intercept))

            y_ = y - slope*x
            ppp = np.percentile( y_, [ PERCENTILE_SECOND ] )
            # print( [i], 'slope=', slope )

            slope_list.append(slope)

            if last_slope is not None:
                diff = abs( slope - last_slope )
                if init_diff is None:
                    init_diff = diff
                if ( diff  < self.very_small_sploe
                    or diff < init_diff * CONVERGENCE_RATIO
                    ):
                    break

            if alternating:
                # break
                pass

            last_slope = slope

        # using a list instead of a tuple for item asignment later
        ret = [xpp, ypp, slope, intercept, ppp, None]

        if self.curve is not None:
            peak_info = self.curve.peak_info
            rightmost_peak_foot_x = int(peak_info[-1][2])
            # print('peak_info=', peak_info, "rightmost_peak_foot_x=", rightmost_peak_foot_x)
            if xpp[0] > rightmost_peak_foot_x:
                if self.logger is not None:
                    self.logger.warning('retrying to replace the suspicious result in LPM')
                slice_ = slice(0, rightmost_peak_foot_x)
                x_ = y[slice_]
                y_ = y[slice_]
                retried_ret = self.get_low_percentile_params(x_, y_, max_iter_num, debug=debug)
                if abs(retried_ret[2]) < abs(ret[2]):
                    if self.logger is not None:
                        self.logger.warning('adopted a retried result in LPM')
                    retried_ret[-1] = slice_
                    return retried_ret

        if debug and self.logger is not None:
            self.logger.info('number of iterations was %d in LPM', i)

        return ret

    def demo_plot( self, title="Demo", parent=None, ecurve=None, p_final=None, hires=False ):
        import molass_legacy.KekLib.DebugPlot as plt
        A, B = self.params
        if p_final is None:
            p_final = PERCENTILE_FINAL

        figsize = (24,12) if hires else (10,8)
        fig = plt.figure(figsize=figsize)

        if hires:
            fontsize    = 48
            legendsize  = 36
            labelsize   = 36
            linewidth   = 5
            markersize  = 20
        else:
            fontsize    = 30
            legendsize  = 20
            labelsize   = 20
            linewidth   = 3
            markersize  = 8

        ax = fig.add_subplot( 111 )
        # ax.set_title( title, fontsize=fontsize )
        ax.plot( self.y, 'o', markersize=markersize )
        # ax.plot( self.y, ':', color='green' )
        ax.plot( self.xpp, self.ypp, 'o', markersize=markersize, color='orange', label='under %d%% points' % PERCENTILE_SECOND )

        baseline_ = A*self.x + B
        ax.plot( baseline_, color='red', linewidth=linewidth, alpha=1, label='LPM result with %d%% offset' % p_final )
        # ax.plot( self.npp, self.y[self.npp], 'o', color='red', label='10% point (default)' )

        ax.tick_params( labelsize=labelsize )
        ax.legend( fontsize=legendsize )
        fig.tight_layout()
        return plt.show()
