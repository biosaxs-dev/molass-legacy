# coding: utf-8
"""

    ElutionBaseCurve.py

    scattering baseline solver assuming Gaussian Curves

    Copyright (c) 2017-2020, SAXS Team, KEK-PF

"""
import numpy                as np
from scipy                  import stats
from bisect                 import bisect_right
# from scipy.interpolate      import UnivariateSpline
from scipy.interpolate      import LSQUnivariateSpline
import logging
from BasePercentileOffset   import base_percentile_offset, DEFAULT_OFFSET
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker

DEBUG   = False

FWHM_SIGMA_RATIO        = 2*np.sqrt(2*np.log(2))
ACCEPTABLE_WIDTH_MIN    = 10
ACCEPTABLE_RATIO_LOWS   = 0.4
DEFAULT_SIZE_SIGMA      = 5
NUM_REGRESSION_POINTS   = 20

def get_spline_smoothing( max_y ):
    adjust = -0.5 if max_y > 0.01 else 0.5
    smoothing = np.power( 10, np.log( max_y ) + adjust )
    return smoothing

def get_spline_smoothing_better( y, max_y ):
    x   = np.arange(NUM_REGRESSION_POINTS)
    slope, intercept, r_value, p_value, std_err = stats.linregress( x, y[0:NUM_REGRESSION_POINTS] )
    s   = std_err**2 * 1e3 * len( y )
    # print( 'std_err=', std_err )
    # print( 'max_y=', max_y, 's=%.1e' % s )
    return s

class ElutionBaseCurve:
    def __init__( self, y ):
        self.y      = y
        self.max_n  = np.argmax( y )
        # self.max_y  = y[self.max_n] - np.min( y )
        self.max_y  = y[self.max_n]
        self.x      = np.arange( len(self.y) )
        self.sorted_y = sorted( y )
        self.height = self.sorted_y[-1] - self.sorted_y[0]
        self.noisiness = None
        self.logger = logging.getLogger( __name__ )

    def compute_noisiness( self ):
        if self.noisiness is None:
            # smoothing = get_spline_smoothing_better( self.y, self.max_y )
            # spline = UnivariateSpline( self.x, self.y, s=smoothing )
            knots = np.linspace( 0, len(self.y), len(self.y)//10 )
            self.spline = LSQUnivariateSpline( self.x, self.y, knots[1:-1] )
            self.ys = self.spline( self.x )
            std_diff = np.std( self.y - self.ys )
            self.noisiness = noisiness = std_diff/self.height

        return self.noisiness

    def recognize_peaks( self ):
        print('recognize_num_peaks')
        ylow, yhigh = np.percentile( self.y, [ 30, 70 ] )
        ipp = self.y < ylow
        x_  = self.x[ipp]
        y_  = self.y[ipp]
        slope, intercept, r_value, p_value, std_err = stats.linregress( x_, y_ )
        line = slope*self.x + intercept
        y   = self.y - line
        ipp = self.y > yhigh
        # print( 'ipp=', ipp )
        peaks = []
        last_b = False
        f = None
        # TODO: faster algorithm;
        for i, b in enumerate( ipp ):
            if b:
                if last_b:
                    continue
                else:
                    f = i
            else:
                if last_b:
                    peaks.append( [ f, i-1 ] )
                else:
                    f = i
            last_b = b

        if last_b:
            peaks.append( [ f, i-1 ] )

        # print( 'peaks=', peaks )
        connected_peaks = []
        last_p = None
        for p in peaks:
            if last_p is not None:
                if p[0] - last_p[1] > 2:
                    connected_peaks.append( last_p )
                else:
                    last_p[1] = p[1]
                    continue

            last_p = p

        connected_peaks.append( last_p )

        # print( 'connected_peaks=', connected_peaks )
        if False:
            from DebugCanvas import DebugCanvas
            def debug_plot( fig ):
                ax = fig.add_subplot( 111 )
                ax.set_title( 'Debug Plot' )
                ax.plot( self.y )

                for p in connected_peaks:
                    slice_ = slice( p[0], p[1]+1 )
                    ax.plot( self.x[slice_], self.y[slice_], color='red' )

            dc = DebugCanvas( "Debug Plot", debug_plot )
            dc.show()

        return connected_peaks

    def split( self, peaks ):
        assert len( peaks ) > 1

        curves = []
        for i, p in enumerate(peaks[1:]):
            prev = peaks[i]
            boundary = ( prev[1] + p[0] )//2
            curve = ElutionBaseCurve( self.y[prev[0]:boundary] )
            curves.append( curve )
        curve = ElutionBaseCurve( self.y[boundary:] )
        curves.append( curve )
        return curves

    def compute_peak_width( self ):
        peaks = self.recognize_peaks()
        if len( peaks ) > 1:
            curves = self.split( peaks )
            widths = []
            for curve in curves:
                w = curve.compute_peak_width_impl()
                # print( 'w=', w )
                if w is not None:
                    widths.append( w )

            # TODO: consider asymmetry
            return np.max( widths ) * len( widths )
        else:
            return self.compute_peak_width_impl()

    def compute_peak_width_impl( self ):
        hm  = self.max_y / 2
        y_L = self.y[0:self.max_n]
        i_L = bisect_right( sorted( y_L ), hm )
        y_R = self.y[self.max_n:]
        j   = bisect_right( sorted( y_R ), hm )
        i_R = len(self.y) - j
        self.i_L    = i_L
        self.i_R    = i_R
        self.width  = i_R - i_L
        self.ret_width  = None

        if self.width < ACCEPTABLE_WIDTH_MIN or i_R == len(self.y):
            return self.ret_width

        hm  = np.min( self.y[[i_L, i_R]] )
        num_lows = np.sum( self.y[i_L:i_R] < hm )
        self.ratio  = num_lows / self.width
        if self.ratio > ACCEPTABLE_RATIO_LOWS:
            return self.ret_width

        self.ret_width  = self.width
        return self.ret_width

    def compute_size_sigma( self ):
        try:
            peak_width  = self.compute_peak_width()
            print( 'peak_width=', peak_width )
            if peak_width is None:
                size_sigma  = DEFAULT_SIZE_SIGMA
            else:
                size_sigma  = len(self.y) * 0.5 / peak_width * FWHM_SIGMA_RATIO
        except:
            size_sigma = 5
            etb = ExceptionTracebacker()
            self.logger.warning( str(etb) + '; size_sigma is set to ' + str(size_sigma) )

        if DEBUG:
            print( 'n=', self.max_n )
            print( 'peak_width=', peak_width  )
            print( 'size_sigma=', size_sigma )

        return size_sigma

    def compute_base_percentile_offset( self, q, size_sigma=None, return_tuple=False, return_simple=False ):
        # See Also BasePercentileOffset.py
        # return DEFAULT_OFFSET

        noisiness   = self.compute_noisiness()
        if return_simple:
            bpo_simple = bisect_right( self.sorted_y, self.height*noisiness )/len( self.y )*30  # 30 = 100 * 0.3
            return bpo_simple

        if size_sigma is None:
            size_sigma  = self.compute_size_sigma()
        bpo = base_percentile_offset( q, noisiness, size_sigma=size_sigma )
        if return_tuple:
            return bpo, noisiness, size_sigma
        else:
            return bpo
