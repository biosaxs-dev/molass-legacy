# coding: utf-8
"""
    AtsasPlotter.py

    Copyright (c) 2016-2020, SAXS Team, KEK-PF
"""
from bisect import bisect_right
import numpy as np
from scipy import stats
# import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from matplotlib.font_manager import FontProperties
from DataUtils import get_in_folder

NUM_POINTS_REG  = 40

class AlmergePlotter:
    def __init__(self):
        self.fixed_font         = FontProperties()
        self.fixed_font.set_family( 'monospace' )

    def plot(self, datafiles, indeces, result, ex_data=None, aq_data=None, max_c=None, title=None, force_overlap_q=None, q_limit=None, descend=True, range_info=None, fig_info=None):

        if force_overlap_q is None:
            overlap_from_max = result.overlap_from_max
        else:
            file = datafiles[indeces[0]]
            array = np.loadtxt( file )
            x = array[:, 0]
            overlap_from_max = bisect_right(x, force_overlap_q)
            x = None

        print( 'forced overlap_from_max=', overlap_from_max )

        if fig_info is None:
            fig, axes_ = plt.subplots(nrows=2, ncols=2, figsize=(22, 12))
        else:
            fig, axes_ = fig_info

        ax0 = axes_[0,0]
        ax1 = axes_[0,1]
        ax2 = axes_[1,1]
        ax3 = axes_[1,0]
        axes = [ax1, ax2, ax3]

        side = ' (descend)' if descend else ' (ascend)'
        fig.suptitle("Comparison against ALMERGE using " + get_in_folder() + side, fontsize=30)

        if title is not None:
            ax1.set_title( title + ' - Input' )
            ax2.set_title( title + ' - Overlapped' )
            ax3.set_title( title + ' - Overlapped (Zoomed)' )

        if range_info is not None:
            ecurve, (f, t) = range_info
            ax0.plot(ecurve.x, ecurve.y, color='orange')
            ymin, ymax = ax0.get_ylim()
            ax0.set_ylim(ymin, ymax)
            p = Rectangle(
                    (f, ymin),      # (x,y)
                    t - f,          # width
                    ymax - ymin,    # height
                    facecolor   = 'cyan',
                    alpha       = 0.2,
                )
            ax0.add_patch( p )

        x = None
        plt_slice = None
        reg_slice = None
        x2 = None
        xr = None
        max_yp  = None
        max_i   = None
        max_k   = None
        min_yp  = None
        min_i   = None
        min_k   = None

        plt_list = []

        for k, i in enumerate(indeces):
            file = datafiles[i]
            array = np.loadtxt( file )
            print( i, file, array.shape,  )

            if x is None:
                x = array[:, 0]
                if q_limit is None:
                    j = len(x)
                    sa_view_q = x[-1]/4
                else:
                    j = bisect_right(x, q_limit)
                    sa_view_q = q_limit/4

                sa_view_xlim = sa_view_q**2
                plt_slice = slice(0, j)
                reg_end = min( j, overlap_from_max + NUM_POINTS_REG )
                reg_slice = slice( overlap_from_max, reg_end )
                x2 = x[plt_slice]**2
                xr  = x2[reg_slice]

            y   = array[plt_slice, 1]
            y_  = np.log(y)
            yr  = y_[reg_slice]
            slope, intercept, r_value, p_value, std_err = stats.linregress( xr, yr )
            yp = slope * xr[0] + intercept
            if max_yp is None or yp > max_yp:
                max_yp  = yp
                max_i   = i
                max_k   = k
            if min_yp is None or yp < min_yp:
                min_yp  = yp
                min_i   = i
                min_k   = k
            plt_list.append( [ y_, yp, slope, intercept ] )

        # print( 'min_i=', min_i, 'max_i=', max_i )

        for k, rec in enumerate(plt_list):
            y_, yp, slope, intercept = rec
            lin_y = slope * xr + intercept
            yp_shift = max_yp - yp

            i = indeces[k]
            if i == max_i:
                c = 'orange'
                alpha = 0.5
                label='max_c'
            elif i == min_i:
                c = 'green'
                alpha = 0.5
                label='min_c'
                min_y = y_ + yp_shift
            else:
                c = 'gray'
                alpha = 0.1
                label=None
            ax1.plot( x2, y_, color=c, alpha=alpha, label=label )
            ax1.plot( xr, lin_y, ':', color='red', alpha=0.5 )
            ax1.plot( xr[0], lin_y[0], 'o', color='red', markersize=5 )
            for ax in axes[1:]:
                ax.plot( x2, y_ + yp_shift, color=c, alpha=alpha, label=label )
                ax.plot( xr[0], lin_y[0] + yp_shift, 'o', color='red', markersize=5 )

        dev_slice = slice( 0, overlap_from_max )
        min_y_slice = min_y[dev_slice]
        zeros = np.zeros(overlap_from_max)

        if ex_data is None:
            ex_data = result.exz_array

        y_ex = np.log( ex_data[plt_slice,1] )
        diff = y_ex[dev_slice] - min_y_slice
        a_pos_dev = np.sum( np.max( [ zeros,  diff ], axis=0 ) )
        a_neg_dev = np.sum( np.max( [ zeros, -diff ], axis=0 ) )
        for ax in axes[1:]:
            ax.plot( x2, y_ex, color='red', label='almerge', alpha=0.5 )

        if aq_data is not None:
            scale = ex_data[overlap_from_max,1]/aq_data[overlap_from_max,1]
            # scale = max_c
            y_aq = np.log( aq_data[plt_slice,1]*scale )
            diff = y_aq[dev_slice] - min_y_slice
            k_pos_dev = np.sum( np.max( [ zeros,  diff ], axis=0 ) )
            k_neg_dev = np.sum( np.max( [ zeros, -diff ], axis=0 ) )
            for ax in axes[1:]:
                ax.plot( x2, y_aq, color='cyan', label='A(q)', alpha=0.5 )

        # plot again to be visible
        ax2.plot( xr[0], max_yp, 'o', color='red', markersize=5 )

        ymin, ymax = ax1.get_ylim()
        yoffset = ( ymax - ymin ) * 0.2

        ax1.annotate( 'Points to be made to overlap', xy=(xr[0], min_yp), fontsize=16,
                                xytext=( xr[0], min_yp - yoffset ),
                                ha='left',
                                arrowprops=dict( headwidth=3, width=0.3, color='gray' ),
                                )

        xmin, xmax = ax2.get_xlim()
        ymin, ymax = ax2.get_ylim()

        def annotate_overlap_point(ax, side):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xmid = (xmax + xmin)/2
            if xr[0] < xmid:
                xoffset = ( xmax + xmin ) * 0.1
                yoffset = ( ymax - ymin ) * 0.05
                ha = 'left'
            else:
                xoffset = -( xmax + xmin ) * 0.1
                yoffset = ( ymax - ymin ) * 0.2
                ha = 'center'

            ax.annotate( 'Point made to overlap', xy=(xr[0], max_yp), fontsize=16,
                                    xytext=( xr[0] + xoffset, max_yp + yoffset ), ha=ha,
                                    arrowprops=dict( headwidth=3, width=0.3, color='gray' ),
                                    )

        xmin3 = xmin/8
        xmax3 = max(sa_view_xlim, xr[0]+abs(xmin3))
        ax3.set_xlim(xmin3, xmax3)
        w = (xmax3 - xmin)/(xmax - xmin)

        sa_view_ylim = min(ymin * w + ymax * (1 - w), max_yp)

        aslice = slice(0,overlap_from_max)
        ploted_max = max(np.max(y_ex[aslice]), np.max(y_aq[aslice]))
        ymax_ = ymax - (ymax - ploted_max)*3/4
        ax3.set_ylim( sa_view_ylim, ymax_ )

        for k, ax in enumerate(axes[1:]):
            annotate_overlap_point(ax, k)

        if False:
            if ex_data is not None and aq_data is not None:
                xmin, xmax = ax3.get_xlim()
                ymin, ymax = ax3.get_ylim()
                yoffset = ( ymax - ymin ) * 0.03

                tx = xmin * 0.75 + xmax * 0.25
                ty = ymin * 0.25 + ymax * 0.75
                ax3.text( tx, ty,           'almerge : pos_dev/neg_dev=%5.3f' % ( a_pos_dev/a_neg_dev ), fontproperties=self.fixed_font )
                ax3.text( tx, ty-yoffset,   'A(q)    : pos_dev/neg_dev=%5.3f' % ( k_pos_dev/k_neg_dev ), fontproperties=self.fixed_font )

        for ax in axes:
            ax.set_xlabel('Q²', fontsize=16)
            ax.set_ylabel('Ln(I)', fontsize=16)
            ax.legend(fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        # plt.show()

    def show(self):
        plt.show()
