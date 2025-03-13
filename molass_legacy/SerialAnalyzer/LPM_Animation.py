# coding: utf-8
"""

    LPM_Animation.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""
import copy
import numpy            as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lmfit              import minimize, Parameters
from scipy              import stats
from molass_legacy._MOLASS.SerialSettings     import get_setting

PERCENTILE_FIRST        = 25
PERCENTILE_SECOND       = 25
PERCENTILE_FINAL        = 10
VERY_SMALL_SLOPE_RATIO  = 0.01
CONVERGENCE_RATIO       = 0.1

def animation_example():
    fig = plt.figure()
    x = np.arange(0, 10, 0.1)

    ims = []
    for a in range(50):
        y = np.sin(x - a)
        line, = plt.plot(x, y, "r")
        ims.append([line])

    ani = animation.ArtistAnimation(fig, ims)
    # ani.save('anim.gif', writer="imagemagick")
    # ani.save('anim.mp4', writer="ffmpeg")
    plt.show()

def animation_example2():
    fig = plt.figure()
    x = np.arange(0, 10, 0.1)

    ims = []
    for a in range(50):
        y = np.sin(x - a)
        im = plt.plot(x, y, "b")
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims)
    ani.save('sample.gif', writer='imagemagick')

class LPM_AnimationDialog:
    pass


class ScatteringBaseline:
    def __init__( self, y, height=None, anim=False ):
        self.y = y
        self.x = x = np.arange( len(self.y) )
        if anim:
            sbl = ScatteringBaseline(y, anim=False)
            sbl.solve(p_final=7)
            n = sbl.npp
            self.fig, self.axes = plt.subplots(ncols=2, figsize=(16,7))
            self.fig.suptitle("Low Percentile Method Animation", fontsize=30)
            for ax in self.axes:
                ax.plot( y, color='gray' )

            ax1, ax2 = self.axes
            xmin, xmax = ax1.get_xlim()
            ymin, ymax = ax1.get_ylim()
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)

            width = xmax - xmin
            height = ymax - ymin
            scale = 0.1

            fx, fy = x[n], y[n]
            zxmin = fx - width*scale/2
            zxmax = fx + width*scale/2
            zymin = fy - height*scale/2
            zymax = fy + height*scale/2

            ax2.set_xlim(zxmin, zxmax)
            ax2.set_ylim(zymin, zymax)

            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.9)
            self.anim_data_lists = [[],[]]

        self.anim = anim
        if height is None:
            height  = np.max( y ) - np.min( y )

        self.very_small_sploe = height / len(y) * VERY_SMALL_SLOPE_RATIO
        print( 'self.very_small_sploe=', self.very_small_sploe )

    def solve( self, p_final=PERCENTILE_FINAL, max_iter_num=10, no_shift=False ):
        xpp, ypp, slope, intercept, ppp = self.get_low_percentile_params( self.x, self.y, max_iter_num )
        print( 'ppp=', ppp )
        y_  = self.y - slope * self.x
        p   = np.percentile( y_, p_final )
        p_  = np.where( y_ <= p )[0]
        m   = np.argmax( y_[p_] )
        n   = p_[m]

        A   = slope
        if no_shift:
            B   = intercept
        else:
            B   = self.y[n] - slope*n

        if self.anim:
            for k, ax in enumerate(self.axes):
                anim_data = self.anim_data_lists[k]
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                tx  = (xmin + xmax)/2
                ty  = (ymin + ymax)/2
                text = ax.text(tx, ty, "Shift", alpha=0.1, fontsize=200, ha="center", va="center")

                def compute_line( a, b ):
                    xp = []
                    yp = []
                    for x_ in self.x[ [0, -1] ]:
                        xp.append( x_ )
                        yp.append( a * x_ + b )
                    return xp, yp

                lpm_points = anim_data[-1][-1]
                point1, = ax.plot(p_, self.y[p_], 'o', color="yellow")
                point2, = ax.plot(n, self.y[n], 'o', color="red", markersize=10)

                xp, yp = compute_line( A, intercept )
                line, = ax.plot(xp, yp, "r")

                anim_data.append( [text, line, lpm_points, point1, point2] )

                if False:
                    xp, yp = compute_line( A, ( intercept + B )/2 )
                    line, = ax.plot(xp, yp, "r")
                    anim_data.append( [text, line, lpm_points, point1, point2] )

                xp, yp = compute_line( A, B )
                line, = ax.plot(xp, yp, "r")

                anim_data.append( [text, line, lpm_points, point1, point2] )

        self.xpp    = xpp
        self.ypp    = ypp
        self.npp    = n
        print('Base point: ', n)
        self.params = ( A, B )
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

    def get_low_percentile_params( self, x, y, max_iter_num ):
        y_  = y
        ppp = np.percentile( y_, [ PERCENTILE_FIRST ] )
        last_slope  = None
        init_diff   = None
        alternating = False
        last_xpp_avg = None
        half_iter_num = max_iter_num//2
        slope_list = []
        x_center = np.average(x)

        for i in range(max_iter_num):
            altered = False
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
                xpp = np.where( y_ <= ppp[0] )[0]
                ypp = y[xpp]
                slope, intercept, r_value, p_value, std_err = stats.linregress( xpp, ypp )

                xpp_avg = np.average(xpp)
                if last_xpp_avg is not None:
                    altered = (xpp_avg - x_center)*(last_xpp_avg - x_center) < 0
                    print([i], last_xpp_avg, xpp_avg, altered)

                if altered:
                    mid_slope = (slope + last_slope)/2
                    print([i], "slope should be modified from %g to %g" % (slope, mid_slope))
                    # slope = mid_slope

                last_xpp_avg = xpp_avg

            if self.anim:
                print( 'frame', i, ppp )

                for k, ax in enumerate(self.axes):
                    anim_data = self.anim_data_lists[k]
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    tx  = (xmin + xmax)/2
                    ty  = (ymin + ymax)/2
                    text = ax.text(tx, ty, str(i), alpha=0.1, fontsize=360, ha="center", va="center")

                    xp = []
                    yp = []
                    for x_ in self.x[ [0, -1] ]:
                        xp.append( x_ )
                        yp.append( slope * x_ + intercept )
                    line, = ax.plot(xp, yp, "r")

                    lpm_points, = ax.plot( xpp, ypp, 'o', color='orange' )

                    anim_data.append( [text, line, lpm_points] )

            y_ = y - slope*x
            ppp = np.percentile( y_, [ PERCENTILE_SECOND ] )

            print( 'slope=', slope )

            if last_slope is not None:
                diff = abs( slope - last_slope )
                if init_diff is None:
                    init_diff = diff
                if ( diff  < self.very_small_sploe
                    or diff < init_diff * CONVERGENCE_RATIO
                    ):
                    break

            last_slope = slope

        return xpp, ypp, slope, intercept, ppp

    def show_animation( self, save_file=None ):
        y = self.y

        for ax in self.axes:
            ax.plot( y )

        ax1, ax2 = self.axes
        xmin, xmax = ax2.get_xlim()
        ymin, ymax = ax2.get_ylim()
        w1 = 0.02
        w2 = 0.98
        xmin_ = xmin*(1-w1) + xmax*w1
        xmax_ = xmin*(1-w2) + xmax*w2
        ymin_ = ymin*(1-w1) + ymax*w1
        ymax_ = ymin*(1-w2) + ymax*w2

        for ax, lw in zip(self.axes, [5, 10]):
            ax.plot([xmin_, xmax_, xmax_, xmin_, xmin_], [ymin_, ymin_, ymax_, ymax_, ymin_], color='red', alpha=0.1, linewidth=lw)

        dual_amin_data = []
        for data1, data2 in zip(*self.anim_data_lists):
            dual_amin_data.append(data1+data2)

        # ani = animation.ArtistAnimation(self.fig, self.anim_data)
        ani = animation.ArtistAnimation(self.fig, dual_amin_data, interval=1000, blit=True, repeat_delay=1000 )

        if save_file is not None:
            print("saving anim to ", save_file)
            ani.save( save_file, writer="ffmpeg" )

        plt.show()
