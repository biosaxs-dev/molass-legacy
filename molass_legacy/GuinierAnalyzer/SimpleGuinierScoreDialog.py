# coding: utf-8
"""
    SimpleGuinierScoreDialog.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy        as np
from scipy          import stats
from molass_legacy.KekLib.OurTkinter     import Tk, Dialog
from CanvasDialog   import CanvasDialog
from SimpleGuinier  import KNOT_NO_ADDITION
from SimpleGuinierScore import ( FWD_GUINIER_SIZE,
                                evaluate_rg_consistency,
                                evaluate_guinier_interval,
                                compute_end_consistency,
                                compute_fwd_consistency )

class SimpleGuinierScoreDialog( CanvasDialog ):
    def __init__( self, parent, title, guinier ):
        self.guinier    = guinier
        self.px         = guinier.px
        self.dx         = guinier.x[1] - guinier.x[0]
        self.x          = guinier.x2
        self.y          = guinier.log_y
        CanvasDialog.__init__(self, title, parent)

    def show( self, plot_title=None ):
        self.plot_title = plot_title
        CanvasDialog.show( self, self.draw, figsize=(12, 5), toolbar=True )

    def draw( self, fig ):
        self.fig = fig
        fig.canvas.mpl_connect( 'button_press_event', self.on_button_press )
        self.point_list = []

        ax1 = fig.add_subplot( 121 )
        ax2 = fig.add_subplot( 122 )
        self.axes = [ ax1, ax2 ]
        self.redraw()

    def redraw( self, clear=False ):
        for i, ax in enumerate( self.axes ):
            if clear:
                ax.cla()
            if self.plot_title is not None:
                ax.set_title( self.plot_title )

            ax.set_xlabel( 'Q²' )
            ax.set_ylabel( 'Ln(I)' )

            ax.plot( self.guinier.x2, self.guinier.log_y, 'o', markersize=3 )
            ax.plot( self.guinier.x2, self.guinier.log_sy, color='green', alpha=0.5 )
            msize = 5 if i == 0 else 8
            ax.plot( self.guinier.kx2, self.guinier.log_ky, 'o', color='red', markersize=msize )
            if self.guinier.added_knot_type != KNOT_NO_ADDITION:
                ax.plot( self.guinier.kx2[1], self.guinier.log_ky[1], 'o', color='pink', markersize=msize )

            if i == 0:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
            else:
                ax.set_xlim( xmin/16, xmax/64 )
                ymax_ = max( self.guinier.guinier_y[0]+0.5, self.guinier.log_py + 0.1 )
                ymax_ = max( ymax_, self.guinier.log_sy[0] + 0.1 )
                ax.set_ylim( self.guinier.guinier_y[1]-1.0, ymax_ )

            ax.plot( self.guinier.px2, self.guinier.log_py, 'o', color='yellow', markersize=msize )

        self.fig.tight_layout()

    def on_button_press( self, event ):

        if event.inaxes != self.axes[1]:
            return

        x, y = event.xdata, event.ydata
        # print( x, y )

        if len(self.point_list) == 2:
            del self.point_list[:]
            self.redraw( clear=True )

        ax = self.axes[1]
        dist = (self.x - x)**2 + (self.y - y)**2
        n = np.argmin( dist )
        point, = ax.plot( self.x[n], self.y[n], 'o', color='red' )
        self.point_list.append( (n, point) )

        if len( self.point_list ) == 2:
            n1 = self.point_list[0][0]
            n2 = self.point_list[1][0]
            i  = np.array( sorted( [n1,n2] ) )
            ax.plot( self.x[i], self.y[i], color='red' )

            tx = np.average( self.x[i] )
            ty = np.average( self.y[i] )
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xoffset = (xmax-xmin)*0.05
            yoffset = (ymax-ymin)*0.05
            if ty > ymin*0.2 + ymax*0.8:
                yoffset *= -1
            elif ty > ymin*0.4 + ymax*0.6:
                yoffset = 0

            text = self.compute_score_text( i )
            ax.annotate( text, xy=( tx, ty ),
                xytext=( tx + 2*xoffset, ty + yoffset ), alpha=0.5,
                arrowprops=dict( headwidth=5, width=1, color='black', shrink=0, ),
                va='center',
                )

        self.mpl_canvas.show()

    def compute_score_text( self, i ):
        start = i[0]
        stop  = i[1]+1
        slice_ = slice( start, stop )
        x_ = self.x[slice_]
        y_ = self.y[slice_]
        slope, intercept, r_value, p_value, stderr = stats.linregress( x_, y_ )
        rg = np.sqrt( -3*slope )
        x  = np.sqrt( x_[-1] )
        size = stop - start
        end_consistency = end_consistency = compute_end_consistency( start, stop, self.x, self.y )
        fwd_consistency = compute_fwd_consistency( start, stop, self.x, self.y )
        score, score_vector, score_array = evaluate_guinier_interval( self.guinier.basic_quality, self.px, self.dx, rg , size, r_value, end_consistency, fwd_consistency, return_vector=True )

        return 'rg=%g\nq*rg=%g\nsize=%d\nsize_score=%g\nr_value=%g\nlinearity_score=%g\nfwd_consistency=%g\nend_consistency=%g\nscore=%g' % (
                    rg, x*rg, size, score_vector[0], r_value, score, fwd_consistency, end_consistency, score )
