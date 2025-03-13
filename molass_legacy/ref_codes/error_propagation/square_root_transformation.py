"""
    square_root_transformation.py

    Copyright (c) 2016, Masatsuyo Takahashi
"""
import numpy as np
import pylab as plt

delta = 1e-2

intervals = map( lambda xi: np.arange( xi[0], xi[-1], delta ), [ [ 0.5, 0.6 ], [ 0.95, 1.05 ], [ 1.4, 1.5 ] ] )

x = np.arange( 0, 3, delta )

y = 2*x - 1

fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot( 1, 1, 1 )
ax.set_title( 'Comparison:  $2x-1$ vs $\sqrt{2x-1}$' )

ax.set_xlim( [ -0.2, 2.2 ] )
ax.set_ylim( [ -0.2, 2.2 ] )

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

xoffset = ( xmax - xmin ) * 0.05
yoffset = ( ymax - ymin ) * 0.05

ax.plot( [ xmin, xmax ], [ 0, 0 ], color='black' )
ax.plot( [ 0, 0 ], [ ymin, ymax ], color='black' )

ax.plot( x, y, color='blue' )

y_ = np.sqrt( 2*x - 1 )

ax.plot( x, y_, color='green' )

for xi in intervals:
    yL = 2*xi - 1
    yS = np.sqrt( yL )
    yZ = np.zeros( ( len(xi) ) )
    xZ = np.zeros( ( len(xi) ) )
    for i, x_ in enumerate( [ xi[0], xi[-1] ] ):
        ax.plot( [ x_, x_ ], [ ymin, ymax ], ':', color='red' )
        ax.plot( [ 0, x_ ], [ yL[-i], yL[-i] ], ':', color='blue' )
        ax.plot( [ 0, x_ ], [ yS[-i], yS[-i] ], ':', color='green' )

    ax.plot( xi, yZ, color='red', linewidth=7, alpha=0.5 )
    ax.plot( xi, yL, color='blue', linewidth=7, alpha=0.5 )
    ax.plot( xZ, yL, color='blue', linewidth=7, alpha=0.5 )
    ax.plot( xi, yS, color='green', linewidth=7, alpha=0.5 )
    ax.plot( xZ, yS, color='green', linewidth=7, alpha=0.5 )

for x_ in [ 0.6, 1, 1.5 ]:
    yy = np.sqrt( 2*x_ - 1 )
    yd = 1/np.sqrt( 2*x_ - 1 )
    text_ = 'slope is %.2g' % yd
    ax.annotate( text_, xy=( x_, yy ),
        xytext=( x_ + xoffset, yy - yoffset ), alpha=0.5,
        arrowprops=dict( headwidth=5, width=1, facecolor='black', shrink=0, alpha=0.5 ),
        )

plt.show()
