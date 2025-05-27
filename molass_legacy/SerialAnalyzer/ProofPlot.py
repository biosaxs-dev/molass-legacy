# coding: utf-8
"""
    ProofPlot.py

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF
"""
import re
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.ticker      import FuncFormatter
from bisect                 import bisect_right
from PlotUtils              import get_wider_range, convert_to_the_level
from molass_legacy.KekLib.OurMatplotlib          import convert_to_fixed_font_latex

DEBUG = False

class Plotter:
    def __init__( self, intensity, title, rg_info_a=None, rg_info_k=None, orig_info=None, q_limit_index=None ):
        print('Plotter: q_limit_index=', q_limit_index)

        self.orig_array = intensity.orig_array
        # self.x      = intensity.X
        # self.y      = intensity.Y
        self.x = self.orig_array[:,0]**2
        self.y = np.log( self.orig_array[:,1] )

        if rg_info_k is None:
            self.sx     = None
            self.sy     = None
        else:
            dx  = 1e-4
            self.sx = np.arange( self.x[0], self.x[-1] + 0.5*dx, dx )
            x_  = np.sqrt( self.sx )
            # y_  = rg_info_k.fit.model( x_ )
            # self.sy = np.log( y_ )
            self.sy = self.y

        self.rg_info_a  = rg_info_a
        self.rg_info_k  = rg_info_k

        self.fig    = fig = plt.figure( figsize=( 24, 8 ) )
        self.ax1    = fig.add_subplot( 1, 3, 1 )
        self.ax2    = fig.add_subplot( 1, 3, 2 )
        self.ax3    = fig.add_subplot( 1, 3, 3 )

        t_ = len( self.sx ) // 8
        self.sxmin  = self.sx[0]
        self.sxmax  = self.sx[t_]
        sy_ = self.sy[ 0:t_+1 ]
        sy__ = sy_[np.isfinite(sy_)]
        self.symin  = np.min( sy__ )
        self.symax  = np.max( sy__ )

        self.ax1.set_title( title )
        self.draw_scatter_and_lines( self.ax1, orig_info=orig_info )

        sxmin_, sxmax_ = get_wider_range( self.sxmin, self.sxmax, 0.2 )
        symin_, symax_ = get_wider_range( self.symin, self.symax, 0.2 )

        self.ax2.set_title( title + ' (zoomed-in)' )
        self.ax2.set_xlim( sxmin_, sxmax_ )
        self.ax2.set_ylim( symin_, symax_ )
        self.draw_scatter_and_lines( self.ax2, curvature=True )

        self.ax3.set_title( title + ' (zoomed-in closer)' )

        sxmin3_, sxmax3_ = sxmin_, sxmax_ 
        if DEBUG: print( '(1) sxmin3_, sxmax3_=', [sxmin3_, sxmax3_] )
        symin3_, symax3_ = symin_, symax_

        if rg_info_a is not None:
            f0, t0  = rg_info_a.From, rg_info_a.To
            y_ = self.y[ f0:t0+1 ]
            sxmin3_, sxmax3_ = get_wider_range( self.x[f0], self.x[t0], 0.1 )
            symin3_, symax3_ = get_wider_range( np.min( y_ ), np.max( y_ ), 0.1 )

        if rg_info_k is not None:
            f2, t2  = rg_info_k.From, rg_info_k.To
            if f2 is None or t2 is None: return
            y_ = self.y[ f2:t2+1 ]
            xmin_, xmax_ = get_wider_range( self.x[f2], self.x[t2], 0.1 )
            y__ = y_[np.isfinite(y_)]
            ymin_, ymax_ = get_wider_range( np.min( y__ ), np.max( y__ ), 0.1 )
            if rg_info_a is None:
                sxmin3_ = xmin_
                sxmax3_ = xmax_
                symin3_ = ymin_
                symax3_ = ymax_
            else:
                sxmin3_ = min( sxmin3_, xmin_ )
                sxmax3_ = max( sxmax3_, xmax_ )
                symin3_ = min( symin3_, ymin_ )
                symax3_ = max( symax3_, ymax_ )
        if DEBUG: print( '(2) sxmin3_, sxmax3_=', [sxmin3_, sxmax3_] )


        sxmin3_, sxmax3_ = get_wider_range( sxmin3_, sxmax3_, 0.25 )
        if DEBUG: print( '(3) sxmin3_, sxmax3_=', [sxmin3_, sxmax3_] )
        symin3_, symax3_ = get_wider_range( symin3_, symax3_, 0.25 )

        self.ax2.plot(  [ sxmin3_, sxmax3_, sxmax3_, sxmin3_, sxmin3_ ],
                        [ symin3_, symin3_, symax3_, symax3_, symin3_ ],
                        ':', color='black' )

        self.ax3.set_xlim( sxmin3_, sxmax3_ )
        self.ax3.set_ylim( symin3_, symax3_ )
        self.draw_scatter_and_lines( self.ax3, curvature=True, marker=True )

        self.get_qrg_min_max( [ rg_info_a, rg_info_k ] )

        if rg_info_a is not None:
            self.draw_guineier( rg_info_a, 'red' )
            # self.plot_q_rg( rg_info_a, 'red' )

        if rg_info_k is not None:
            self.draw_guineier( rg_info_k, 'cyan' )
            # self.plot_q_rg( rg_info_k, 'cyan' )

        if q_limit_index is not None:
            # fix ax1 limits
            if q_limit_index < self.orig_array.shape[0]:
                xmin1, xmax1 = self.ax1.get_xlim()
                self.ax1.set_xlim( xmin1, xmax1 )
                ymin1, ymax1 = self.ax1.get_ylim()
                self.ax1.set_ylim( ymin1, ymax1 )

                # draw
                q = self.orig_array[q_limit_index,0]
                x = q**2
                self.ax1.plot( [ x, x ], [ ymin1, ymax1 ], color='gray' )

    def x_formatter( self, val, pos ):
        if pos is None: return ''

        if val >= 0:
            index = bisect_right( self.x, val )
            return '%g\n(%g)' % ( val, index )
        else:
            return '%g' % val

    def draw_scatter_and_lines( self, ax, curvature=False, marker=False, orig_info=None ):
        if orig_info is None:
            x_  = self.x
            y_  = self.y
        else:
            sd, n =orig_info
            x   = sd.intensity_array[n,:,0]
            y   = sd.intensity_array[n,:,1]
            positive = y > 0
            x_  = x[positive]**2
            y_  = np.log( y[positive] )

        ax.scatter( x_, y_, s=16 )

        if orig_info is not None:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim( ymin, ymax )
            q_index = sd.get_usable_q_limit()
            if q_index is not None:
                x_  = x[q_index]**2
                ax.plot( [ x_, x_ ], [ ymin, ymax ], color='gray' )

        marker_ = 'x' if marker else None
        # ax.plot( self.sx, self.sy, color='pink', marker=marker_, label='Guinier-Porod fitted curve' )

        ax.set_xlabel( 'Q**2' )
        ax.set_ylabel( 'Ln( I )' )
        ax.xaxis.set_major_formatter( FuncFormatter( self.x_formatter ) )

        sxmin   = self.sxmin
        sxmax   = self.sxmax
        symin   = self.symin
        symax   = self.symax
        ax.plot(    [ sxmin, sxmax, sxmax, sxmin, sxmin ],
                    [ symin, symin, symax, symax, symin ],
                    ':', color='black' )

        if curvature:
            self.draw_curvature_etc( ax, marker=marker )

    def draw_curvature_etc( self, ax, marker=False ):
        axmin_, axmax_ = ax.get_xlim()
        aymin_, aymax_ = ax.get_ylim()

        xoffset = ( axmax_ - axmin_ ) * 0.05
        yoffset = ( aymax_ - aymin_ ) * 0.05

        # Guinier-Porod model boundaries
        if self.rg_info_k is not None and self.rg_info_k.fit.Rg is not None:
            # Rg, d = self.rg_info_k.fit.Rg, self.rg_info_k.fit.degree
            Rg, d = 99, 3
            for d in range( 1, 5 ):
                q = 1/Rg * np.sqrt( 3*d/2 )
                x = q**2
                label_ = 'Guinier-Porod Model boundaries' if d == 1 else None
                ax.plot( [ x, x ], [ aymin_, aymax_ ], '-', color='yellow', label=label_ )

        if self.rg_info_k is not None:
            q1  = self.rg_info_k.fit.q1
            x1  = q1**2
            x_  =  np.array( [q1] )
            y_  = np.log( self.rg_info_k.fit.model( x_ ) )
            y1  = y_[0]
            text_ = 'q1=%.3g, q1*Rg=%.3g' % ( q1, q1 * self.rg_info_k.fit.Rg )
            ax.annotate( text_, xy=( x1, y1 ),
                xytext=( x1 + 2*xoffset, y1 - yoffset ), alpha=0.5,
                arrowprops=dict( headwidth=5, width=1, color='black', shrink=0 ),
                )

    def draw_guineier( self, rec, color ):
        vstrs = []
        for v in [ rec.Rg, rec.Rg_stdev ]:
            vstr = 'NA' if v is None else '%.2f' % v
            vstrs.append( vstr )

        label = '%s: Rg=%s, Rg.stdev=%s' % ( rec.type, vstrs[0], vstrs[1] )

        for i, ax in enumerate( [ self.ax1, self.ax2, self.ax3 ] ):
            label_ = convert_to_fixed_font_latex( label )
            self.plot_guineier_inverval( ax, rec, color, label=label_ )

    def plot_guineier_inverval( self, ax, rec, color, label=None ):
        x = self.x
        y = self.y

        Rg, I0, f, t, a_, b_ = rec.Rg, rec.I0, rec.From, rec.To, rec.a, rec.b

        # print( 'plot_guineier_inverval:', a_,  - ( Rg**2 / 3 ), b_, np.log( I0 ) )
        if Rg is None: return

        if not np.allclose( [a_, b_], [ -( Rg**2 / 3 ), np.log( I0 ) ] ):
            print( 'WARNINIG: not allclose!', [a_, b_], [ -( Rg**2 / 3 ), np.log( I0 ) ] )

        b_ = np.log( I0 )
        a_ = - ( Rg**2 / 3 )
        y0 = b_ + a_ * x[f]
        y1 = b_ + a_ * x[t]

        ax.plot( [ x[f], x[t] ], [ y0, y1 ], marker='o', color=color, label=label )

    def get_qrg_min_max( self, rec_array ):
        self.sqrt_x = np.sqrt( self.x )

        qrg_array = [ 0, 2.46 ]

        for rec in rec_array:
            if rec is None: continue
            if rec.Rg is None: continue
            Rg, f, t = rec.Rg, rec.From, rec.To
            for x in self.sqrt_x[f:t+1]:
                qrg_array.append( x * Rg )

        self.qRg_min = np.min( qrg_array )
        self.qRg_max = np.max( qrg_array )

    def plot_q_rg( self, rec, color ):
        # print( 'plot_q_rg: color=', color, ', rec=', rec )

        ax = self.ax3
        label = '%s: q*Rg' % rec.type
        x = self.sqrt_x

        Rg, f, t = rec.Rg, rec.From, rec.To
        if Rg is None: return

        x_ = x[f:t+1]
        qRg =  x_ * Rg
        ymin, ymax = ax.get_ylim()
        qRg_limits = [ 0.8, 1.3 ]
        qRg_ = np.array( list( qRg ) + qRg_limits + [ 0, 1.7 ] )
        qRg_min = self.qRg_min
        qRg_max = self.qRg_max
        qRg_plot_max = 0.5
        y_ = convert_to_the_level( qRg, ymin, ymax, 0.0, qRg_plot_max, vmin=qRg_min, vmax=qRg_max )
        x__ = x_ ** 2
        label_ = convert_to_fixed_font_latex( label )
        ax.plot( x__, y_, ':',  color=color, label=label_ )

        xmin, xmax = ax.get_xlim()
        xoffset = ( xmax - xmin ) * 0.05
        yoffset = ( ymax - ymin ) * 0.05

        qRg_limits_ = convert_to_the_level( qRg_limits, ymin, ymax, 0.0, qRg_plot_max, vmin=qRg_min, vmax=qRg_max )

        for k, limit in enumerate( qRg_limits_ ):
            # print( 'plot_q_rg: color=', color, ', limit=', limit )
            ax.plot( [ xmin, xmax ], [ limit, limit ], ':', color='black' )
            text_ = 'q*Rg=%g' % qRg_limits[k]
            ax.annotate( text_, xy=( xmin, limit ),
                            xytext=( xmin + xoffset, limit + yoffset ),
                            color='black', alpha=0.5,
                            arrowprops=dict( headwidth=5, width=1, color='black', shrink=0.05 ),
                            )

        y__ = convert_to_the_level( [ qRg[0], qRg[-1] ], ymin, ymax, 0.0, qRg_plot_max, vmin=qRg_min, vmax=qRg_max )
        for i, k in enumerate( [ 0, -1 ] ):
            x__ = x_[k]**2
            text_ = 'q*Rg=%.3g' % qRg[k]
            xoffset_ = xoffset if i==0 else -xoffset*2
            sign_ = 1 if color == 'red' else -1
            yoffset_ = sign_ * ( yoffset if i==0 else -yoffset )
            color_ = color if color == 'red' else 'black'
            ax.annotate( text_, xy=( x__, y__[i] ),
                            xytext=( x__ + xoffset_, y__[i] + yoffset_ ),
                            color=color_, alpha=0.5,
                            arrowprops=dict( headwidth=5, width=1, color=color, shrink=0.05 ),
                            )

    def show_prepare( self ):
        # this is to avoid plt.show() when used with tkinter.
        for ax in [ self.ax1, self.ax2, self.ax3 ]:
            ax.legend()
        # suppressed for the latest version of matplotlib
        # self.fig.tight_layout()

    def show( self, block=True ):
        self.show_prepare()
        plt.show( block )

