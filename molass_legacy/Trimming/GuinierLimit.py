"""
    Trimming.GuinierLimit.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import numpy as np
import logging
from scipy import stats
from molass_legacy.SerialAnalyzer.AveragingDiscussion import extrapolate
from molass_legacy.GuinierAnalyzer.SimpleGuinierScore import compute_rg
from molass_legacy._MOLASS.SerialSettings import get_setting

RG_EST_ELUTION_WIDTH    = 50
RG_EST_ANGLE_WIDTH      = 20

class GuinierLimit:
    def __init__(self, matrix_data, e_curve, pre_rg, qlimit):
        self.logger = logging.getLogger( __name__ )
        self.matrix_data = matrix_data
        self.e_curve = e_curve
        self.pre_rg = pre_rg
        self.qlimit = qlimit
        self.limit = None

    def determine_safe_start(self, debug=False):
        sg = self.pre_rg.sg
        x = sg.x
        y = sg.y
        q_limit = self.qlimit
        stop = len(y) if q_limit is None else q_limit
        n = np.argmax(y[0:stop])
        q_limit = min(n, stop)
        if q_limit > 0:
            m = np.argmin(y[0:q_limit])
        else:
            m = 0
        if m > 1:
            regress_values = []
            for slice_ in [slice(0,m), slice(m,None)]:
                slope, _, r_value = stats.linregress( x[slice_], y[slice_] )[0:3]
                regress_values.append((slope, r_value))
            regress_values = np.array(regress_values)
            min_r_value = np.min(np.abs(regress_values[:,1]))
            if min_r_value > 0.6:
                safe_start = m
                self.logger.info("safe_start is determined to be at Q[%d]=%.3g", m, x[m])
            else:
                safe_start = 0
        else:
            safe_start = 0

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
            plt.push()
            fig, axes = plt.subplots(ncols=3, figsize=(21,7))
            fig.suptitle('determine_safe_start debug for ' + get_in_folder(), fontsize=30)
            for ax, slice_ in zip([axes[0], axes[2]], [slice(None, None), slice(0, self.qlimit)]):
                ax.set_yscale('log')
                ax.plot(x[slice_], y[slice_])
            ax2 = axes[2]
            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim(ymin, ymax)
            x_ = x[safe_start]
            ax2.plot([x_, x_], [ymin, ymax], ':', color='yellow', linewidth=3)
            if m > 1:
                print('regress_values=', regress_values)
                print('min_r_value=', min_r_value)

            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.show()
            plt.pop()

        return safe_start

    def compute_extrapolated_curve(self, safe_start, debug=False):
        default_eno = self.pre_rg.default_eno
        ewidth  = RG_EST_ELUTION_WIDTH//2
        estart  = max(0, default_eno - ewidth)
        estop   = min(self.matrix_data.shape[1], default_eno + ewidth)
        eslice  = slice(estart, estop)

        sg = self.pre_rg.sg
        guinier_start = sg.guinier_start
        aslice  = slice(safe_start, guinier_start + RG_EST_ANGLE_WIDTH*2)

        M   = self.matrix_data[aslice, eslice]

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.DataStructure.MatrixData import simple_plot_3d
            from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
            plt.push()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(get_in_folder())
            simple_plot_3d(ax, M)
            plt.show()
            plt.pop()

        c   = self.e_curve.y[eslice]
        P   = extrapolate(M, c)
        y   = P[:,0]
        return y

    def get_safe_check_stop(self, y):
        width = 5
        stop = None
        for i in range(10, -1, -1):
            i_ = i + 1
            y_ = y[i_:i_+width]
            height = np.max(y_) - np.min(y_)
            ay = np.average(y_)
            ratio = abs(y[i] - ay)/height
            # print([i], "ratio=", ratio)
            if ratio > 2:
                # as with y[0] in 20230303/HasA 
                stop = i
                break
        if stop is None:
            stop = -1
        return stop

    def get_limit(self, acceptable_consistency, debug=False, fig_file=None, debug_auto=False):
        assert acceptable_consistency >= 0 and acceptable_consistency <= 1

        if acceptable_consistency == 0:
            return 0

        safe_start = self.determine_safe_start(debug=debug)
        zx_y = self.compute_extrapolated_curve(safe_start, debug=debug)
        log_zx_y = np.log(zx_y)

        sg = self.pre_rg.sg
        guinier_start = sg.guinier_start

        log_y = sg.log_y
        x2 = sg.x2

        acceptable_ratio = -np.log(acceptable_consistency)
        """
            i.e.,
            np.exp(-acceptable_ratio) == acceptable_consistency
        """

        def compute_extrapolated_rg(i):
            slice_ = slice(i, i+RG_EST_ANGLE_WIDTH)
            x_ = x2[slice_]
            y_ = log_zx_y[slice_]
            # slope, intercept, r_value, p_value, std_err = stats.linregress( x_, y_ )
            slope = stats.linregress( x_, y_ )[0]
            rg = compute_rg(slope)            
            return rg

        limit = i = guinier_start
        reliable_rg = compute_extrapolated_rg(i)
        check_stop = self.get_safe_check_stop(log_zx_y)

        for i in range(guinier_start+1, check_stop, -1):
            rg = compute_extrapolated_rg(i)
            # print([i], rg, diff_ratio)
            if rg > 0:
                diff_ratio = abs(rg - reliable_rg)/reliable_rg
                if diff_ratio <= acceptable_ratio:
                    limit = i
                else:
                    continue
            else:
                continue

        if debug:
            import time
            print("limit=", limit)
            i = guinier_start

            with plt.Dp():
                fig, axes = plt.subplots(ncols=3, figsize=(21, 7))
                ax0, ax1, ax2 = axes

                in_folder = get_setting('in_folder')
                consistency_text = ' with acceptable Rg-consistancy %g' % acceptable_consistency
                fig.suptitle(in_folder + consistency_text, fontsize=20)

                ax0.set_title("Log plot of the whole region", fontsize=16)
                ax0.set_yscale('log')
                ax0.plot(sg.x, sg.y)
                ax0.plot(sg.x[i], sg.y[i], 'o', color='red')

                ax1.set_title("Guinier plot of the whole region", fontsize=16)
                ax1.plot(x2, log_y)
                aslice = slice(safe_start, safe_start+len(zx_y))
                ax1.plot(x2[aslice], log_zx_y)

                xmin1, xmax1 = ax1.get_xlim()
                ymin1, ymax1 = ax1.get_ylim()

                if True:
                    dx = (xmax1 - xmin1)*0.0005
                    dy = (ymax1 - ymin1)*0.005
                    gx = x2[i]
                    gy = log_y[i]
                    # lower = max(0, i - RG_EST_ANGLE_WIDTH)
                    upper = min(len(x2)-RG_EST_ANGLE_WIDTH*2-1, i + RG_EST_ANGLE_WIDTH*2)
                    xmin2 = x2[0] - dx
                    xmax2 = x2[upper] + dx
                    ymin2 = min(log_y[limit], log_y[upper]) - dy
                    ymax2 = max(gy, np.max(log_zx_y)) + dy
                else:
                    xmin2 = xmin1*0.96 + xmax1*0.04
                    xmax2 = xmin1*0.92 + xmax1*0.08
                    ymin2 = ymin1*0.15 + ymax1*0.85
                    ymax2 = ymin1*0.02 + ymax1*0.98

                ax2.set_title("Guinier plot of the small angle region", fontsize=16)
                ax2.set_xlim(xmin2, xmax2)
                ax2.set_ylim(ymin2, ymax2)

                ax2.plot(x2, log_y, label='measured curve')
                ax2.plot(x2[aslice], log_zx_y, label='extrapolated curve')

                for k, ax in enumerate(axes[1:]):
                    ax.plot(x2[i], log_y[i], 'o', color='red', label='guinier start')
                    if k > 0:
                        x_ = x2[i]
                        ax.plot([x_, x_], [ymin2, ymax2], ':', color='red', linewidth=1)
                        i = limit
                        x_ = x2[i]
                        ax.plot(x_, log_y[i], 'o', color='yellow', label='possible start')
                        ax.plot(x_, log_zx_y[i], 'o', color='yellow')
                        ax.plot([x_, x_], [ymin2, ymax2], ':', color='yellow', linewidth=3)

                ax2.legend(fontsize=16)

                fig.tight_layout()
                fig.subplots_adjust( top=0.88 )
                plt.show(block=not debug_auto)
                if fig_file is not None:
                    fig.savefig( fig_file )

            if debug_auto:
                time.sleep(0.5)

        # limit =  int( guinier_start * probability )     # this is a dummy setting
        return limit
