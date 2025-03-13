# coding: utf-8
"""
    GuinierProofPlot.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""
import sys
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.ticker      import FuncFormatter
from bisect                 import bisect_right

class GuinierProofPlot:
    def __init__( self, fit, interval, gunier, optimal_interval_q=None, title=None, ):

        x_  = interval.x
        self.gunier = gunier

        x = x_**2

        y1  = np.log( interval.y )
        y2  = np.log( fit.model( x_, sigma=0 ) )
 
        self.fig    = fig   = plt.figure( figsize=( 24, 8 ) )
        self.ax1    = ax1   = fig.add_subplot( 1, 3, 1 )
        self.ax2    = ax2   = fig.add_subplot( 1, 3, 2 )
        self.ax3    = ax3   = fig.add_subplot( 1, 3, 3 )

        # for ax in [ ax1, ax2 ]: ax.set_axis_bgcolor('lightgray')

        if title is not None:
            ax1.set_title( title )

        ax2.set_title( 'Plot for Anomaly Index' )
        ax3.set_title( 'Curvature(s)' )

        for ax in [ax1, ax2]:
            ax.plot( x, y1, 'o' )
            ax.xaxis.set_major_formatter( FuncFormatter( self.x_formatter ) )

            ax.plot( x, y2, color='pink', label='Guinier-Porod model fitted' )

            if interval.smoother is not None:
                if interval.smoother.is_guinier:
                    y3  = interval.smoother( x )
                else:
                    y3  = np.log( interval.smoother( x_ ) )
                ax.plot( x, y3, color='cyan', label='Smooth line on the interval' )

        ymin1, ymax1 = ax1.get_ylim()
        ax1.set_ylim( [ ymin1, ymax1 ] )

        for d in range(1,5):
            q = 1 / fit.Rg * np.sqrt( 3*d/2 )
            q2 = q**2
            for ax in [ ax1, ax2 ]:
                ax.plot( [ q2, q2 ], [ ymin1, ymax1 ], ':', color='orange' )

        if optimal_interval_q is not None:
            for q in optimal_interval_q:
                q2 = q**2
                for ax in [ ax1, ax2 ]:
                    ax.plot( [ q2, q2 ], [ ymin1, ymax1 ], ':', color='green' )

        if gunier.sx_array is not None:
            colors = [ 'yellow', 'red' ]
            markers = [ 'x', '' ]
            labels  = [ 'Spline for IpI Check', 'Spline for Aggregation Check' ]
            for i in range( 1 ):
                sx = gunier.sx_array[i]
                sy = gunier.sy_array[i]
                # print( i, 'len(sx)=', len(sx) )
                ax2.plot( sx, sy, marker=markers[i], color=colors[i], label=labels[i] )
                if i == 0:
                    cy = gunier.cy_array[i]
                    ax3.plot( sx, cy, color='green' )

            xmin2 = 0.0
            xmax2 = np.max( sx ) + 0.0001
            ymin2 = np.min( sy ) - 0.1
            ymax2 = np.max( sy ) + 0.1
            ax2.set_xlim( xmin2, xmax2 )
            ax2.set_ylim( ymin2, ymax2 )

            ax1.plot(   [ xmin2, xmax2, xmax2, xmin2, xmin2 ],
                        [ ymin2, ymin2, ymax2, ymax2, ymin2 ],
                        ':', color='black' )

        ax1.legend()
        ax2.legend()
        plt.tight_layout()

    def x_formatter( self, val, pos ):
        if pos is None: return ''

        if val >= 0:
            index = bisect_right( self.gunier.x, val )
            return '%g\n(%g)' % ( val, index )
        else:
            return '%g' % val

    def show( self, block=False ):
        plt.show( block=block )

    def close( self ):
        plt.close()
