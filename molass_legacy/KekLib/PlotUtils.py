# coding: utf-8
"""
    PlotUtils.py

    Copyright (c) 2016-2020, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np

def get_wider_range( vmin, vmax, ratio ):
    vmin_   = vmin * ( 1 + ratio ) + vmax * ( -ratio )
    vmax_   = vmin * ( -ratio ) + vmax * ( 1 + ratio )
    return vmin_, vmax_

def convert_to_the_level( yval, ymin, ymax, posf, post, vmin=None, vmax=None ):
    # print( 'convert_to_the_level: vmin, vmax=', vmin, vmax )
    if vmin is None:
        vmin = np.min( yval )
    if vmax is None:
        vmax = np.max( yval )
    ymin_ = ymin * ( 1 - posf ) + ymax * posf
    ymax_ = ymin * ( 1 - post ) + ymax * post
    ratio = ( ymax_ - ymin_ ) / ( vmax - vmin )
    return ( yval - vmin ) * ratio + ymin_

def plot_line( ax, x, y, rec, color, label=None, plot_qrg=False ):
    Rg, a, b, f, t = rec

    a_ = - ( Rg**2 / 3 )
    y0 = b + a_ * x[f]
    y1 = b + a_ * x[t]

    print( 'plot_line: Rg=', Rg, '; a=', a )
    print( [ x[f], x[t] ], [ y0, y1 ] )

    ax.plot( [ x[f], x[t] ], [ y0, y1 ], marker='o', color=color, label=label )

"""
    https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
"""
def align_yaxis_np(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:,1] / (extrema[:,1] - extrema[:,0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0,1] = extrema[0,0] + tot_span * (extrema[0,1] - extrema[0,0])
    extrema[1,0] = extrema[1,1] + tot_span * (extrema[1,0] - extrema[1,1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]
