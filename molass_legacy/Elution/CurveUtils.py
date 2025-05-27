"""
    Elution.CurveUtils.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import logging
from bisect import bisect_right
import numpy as np
import matplotlib.patches as mpl_patches      # 'as patches' does not work properly
from itertools import combinations
import molass_legacy.KekLib.OurStatsModels as sm
from molass_legacy._MOLASS.SerialSettings import get_setting, get_xray_picking
from molass_legacy.KekLib.SciPyCookbook import smooth

PEAK_FIND_ALLOW     = 2.0       # > abs( 418.5 - 416.85 ) for 20180225

def is_included_in( x, x_list ):
    found = False
    for x_ in x_list:
        if abs( x_ - x ) < PEAK_FIND_ALLOW:
            found = True
            break
    return found

def get_around_slice( vector, val, num ):
    num_half    = num//2
    num_r       = num%2
    i = bisect_right( vector, val )
    return max( 0, i - num_half ), min( len(vector), i + num_half + num_r )

def get_xray_elution_vector( qvector, xray_array ):
    intensity_picking       = get_xray_picking()
    num_points_intensity    = get_setting( 'num_points_intensity' )
    slice_params = get_around_slice( qvector, intensity_picking, num_points_intensity )
    slice_ = slice( slice_params[0], slice_params[1] )
    # print( '===== slice_params=', slice_params, 'qvector[slice_]=', qvector[slice_] )
    return np.average( xray_array[:, slice_, 1], axis=1 ), slice_

def get_probably_corresponding_index( a_curve, x_curve ):
    a_size  = len(a_curve.x)
    x_size  = len(x_curve.x)

    a_peaks = np.array( [ info[1] for info in a_curve.peak_info ] )
    x_peaks = np.array( [ info[1] for info in x_curve.peak_info ] )

    num_a   = len(a_peaks)
    num_x   = len(x_peaks)
    index_a = list(range(num_a))
    index_x = list(range(num_x))

    if num_a == num_x:
        return [ index_a, index_x ], None

    if False:
        from CanvasDialog   import CanvasDialog

        def plot_func( fig ):
            ax1 = fig.add_subplot( 121 )
            ax2 = fig.add_subplot( 122 )

            ax1.plot( a_curve.y )
            ax2.plot( x_curve.y )

            def plot_peak( ax, curve ):
                for info in curve.peak_info:
                    peak = info[1]
                    x   = peak
                    y   = curve.spline( x )
                    ax.plot( x, y, 'o', color='red' )

            plot_peak( ax1, a_curve )
            plot_peak( ax2, x_curve )

            fig.tight_layout()
        dialog = CanvasDialog( "Debug: get_probably_corresponding_index", adjust_geometry=True )
        dialog.show( plot_func, figsize=(16, 8) )
        if not dialog.applied:
            from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
            debug_curve = ElutionCurve( x_curve.y, debug_plot=True )

    if num_a > num_x:
        reversed_ = False
        peaks1  = a_peaks
        peaks2  = x_peaks
        index1  = index_a
        index2  = index_x
        num1    = num_a
        num2    = num_x
        size1   = a_size
        size2   = x_size
    else:
        reversed_ = True
        peaks1  = x_peaks
        peaks2  = a_peaks
        index1  = index_x
        index2  = index_a
        num1    = num_x
        num2    = num_a
        size1   = x_size
        size2   = a_size

    def evaluate_mapping( index ):
        y   = [ 0, size1-1 ]
        x   = [ 0, size2-1 ]
        w   = [ 1, 1 ]
        for p1, p2 in zip( peaks1[index], peaks2 ):
            y.append( p1 )
            x.append( p2 )
            w.append( 10 )

        X   = sm.add_constant(x)
        mod = sm.WLS( y, X, weights=w )
        res = mod.fit()
        return res

    # find minimum ssr result
    min_result  = None
    min_index   = None
    for c in combinations(index1, num2):
        index = list(c)
        res = evaluate_mapping( index )
        if min_result is None or res.ssr < min_result.ssr:
            min_result  = res
            min_index   = index

    intercept, slope = min_result.params
    if reversed_:
        return (index2, min_index), ( 1/slope, -intercept/slope )
    else:
        return (min_index, index2), ( slope, intercept )

SMOOTHING_WINDOW_LEN    = 10

def smoothed_curve_y(curve, y=None):
    if y is None:
        y = curve.y
    try:
        num_points = int(curve.get_primary_peak_num_points()/4) + 2
    except:
        # 20170426 why?
        logger = logging.getLogger(__name__)
        logger.warning("curve.get_primary_peak_num_points() failed.")
        num_points = SMOOTHING_WINDOW_LEN
    # num_points = 5 for 20190309_3
    # print('num_points=', num_points)
    window_len = min(SMOOTHING_WINDOW_LEN, num_points)
    sy = smooth(y, window_len=window_len)
    return sy

def make_smoothed_y(y, window_len=SMOOTHING_WINDOW_LEN):
    approximate_peak_num_points = len(y)/8
    num_points = int(approximate_peak_num_points/4) + 2
    # num_points = 5 for 20190309_3
    # print('num_points=', num_points)
    window_len = min(window_len, num_points)
    sy = smooth(y, window_len=window_len)
    return sy

MAJOR_PEAK_RATIO        = 0.03

def proof_plot( curve, parent, fig, in_folder=None, data_type='', fontsize=16, feature=False ):
    # TODO: refactoring to remove parent

    if in_folder is not None:
        elution = 'Elution' if data_type == '' else ' elution'
        fig.suptitle( data_type + elution + ' curve from folder ' + in_folder, fontsize=fontsize )

    ax1  = fig.add_subplot( 121 )
    ax2  = fig.add_subplot( 122 )

    ax1.set_title( "Recognized peaks", fontsize=fontsize )
    ax2.set_title( "Derivatives and roots for major peaks", fontsize=fontsize )

    x   = curve.x
    y   = curve.y
    spline  = curve.spline
    d1  = curve.d1
    d2  = curve.d2
    yˈ  = curve.d1y
    yˈˈ = d2(x)

    ptx = curve.peak_top_x
    ranges = curve.peak_info
    boundaries = curve.boundaries
    feet = np.array([[r[0], r[2]] for r in ranges]).flatten()

    ax1.plot( x, y, label='data' )
    ax1.plot( x, spline(x), label='spline' )
    ax1.plot( x, curve.feature_y, ':', label='feature_y' )
    # iknots = curve.knots[1:-1]
    # ax1.plot( iknots, spline(iknots), 'o', color='green', label='knots' )

    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim(xmin, xmax)
    y_ = curve.min_y + curve.height * MAJOR_PEAK_RATIO
    ax1.plot( [xmin, xmax], [y_, y_], ':', color='red', label='peak recognition limit' )
    ax1.plot( ptx, spline(ptx), 'o', color='red', label='recongnized peak tops' )
    ax1.plot( feet, spline(feet), 'o', color='yellow', label='foot points' )

    if len(boundaries) > 0:
        ax1.plot( boundaries, spline(boundaries), 'o', color='green', label='recongnized peak boundaries' )

    if feature:
        ax1.plot( curve.feature.roots, spline(curve.feature.roots), 'o', color='yellow', label='inflections' )

    ax2.plot( x, yˈ, label='1st derivative' )
    ax2.plot( x, yˈˈ, label='2nd derivative' )
    ax2.plot( ptx, d1(ptx), 'o', color='red', label='recongnized peak tops' )
    if feature:
        ax2.plot( curve.feature.roots, d1(curve.feature.roots), 'o', color='yellow', label='inflections' )

    ymin, ymax = ax1.get_ylim()
    for range_ in ranges:
        f = range_[0]
        t = range_[2]
        p = mpl_patches.Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.1,
            )
        ax1.add_patch( p )

    ax1.legend(fontsize=fontsize)
    ax2.legend(fontsize=fontsize)
    fig.tight_layout()
    if in_folder is not None:
        fig.subplots_adjust( top=0.88 )

def simple_plot(ax, curve, title=None, legend=True, color=None, major_peaks=False, boundaries=None, spline=False ):
    x = curve.x
    y = curve.y
    if major_peaks:
        top = [ info[1] for info in curve.get_major_peak_info() ]
        bnd = curve.get_major_valley_bottoms(top)
    else:
        top = curve.peak_top_x
        bnd = curve.boundaries if boundaries is None else boundaries
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.plot(x, y, color=color, label='data')
    ax.plot(top, curve.spline(top), 'o', color='red', label='peak tops')
    if len(bnd) > 0:
        ax.plot(bnd, curve.spline(bnd), 'o', color='green', label='valley bottoms')

    if spline:
        ax.plot(x, curve.sy, ':', color='green', label='spline')

    if legend:
        ax.legend()

# TODO: remove duplication of this code
def rotate( th, wx, wy ):
    k = -1 if th > 0 else 0
    cx  = wx[k]
    cy  = wy[k]
    wx_ = wx - cx
    wy_ = wy - cy
    c   = np.cos( th )
    s   = np.sin( th )
    return cx + ( wx_*c - wy_* s ), cy + ( wx_*s + wy_* c )

MIN_NUM_POINTS  = 5

def find_rotated_extreme_arg(curve, start, stop, extremun_flag):
    slice_ = slice(start, stop)

    x = curve.x[slice_]
    y = curve.sy[slice_]    # use the smoothed

    assert len(x) >= MIN_NUM_POINTS

    x1, x2 = x[[0, -1]]
    y1, y2 = y[[0, -1]]

    xscale = 1/abs(x2 - x1)
    yscale = 1/abs(y2 - y1)

    x_ = x * xscale
    y_ = y * yscale

    y_end_ratio = (y_[-1] - y_[0])/curve.height     # better be nearest peak height

    if abs(y_end_ratio) < 0.5:
        rx, ry = x_, y_
    else:
        rotate_sign = -1 if y_end_ratio > 0 else 1
        angle = rotate_sign* np.pi/4
        rx, ry = rotate(angle, x_, y_)

    if extremun_flag < 0:
        pnn = np.argpartition(ry, MIN_NUM_POINTS)
        n = int(np.average(pnn[0:5]) + 0.5)
    else:
        k = len(ry) - MIN_NUM_POINTS
        pnn = np.argpartition(ry, k)
        n = int(np.average(pnn[k:]) + 0.5)

    ret_n = start + n

    if False:
        import molass_legacy.KekLib.DebugPlot as plt

        plt.push()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        ax1, ax2 = axes

        fig.suptitle("find_rotated_extreme_arg")
        simple_plot(ax1, curve)
        ax1.plot(curve.x[ret_n], curve.y[ret_n], 'o', color='yellow')

        x_ = curve.x[start:stop]
        y_ = curve.y[start:stop]
        ax2.plot(rx, ry)
        ax2.plot(rx[n], ry[n], 'o', color='yellow')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()
        plt.pop()

    return ret_n
