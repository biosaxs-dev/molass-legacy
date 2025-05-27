"""
    ScatteringBasesurface.py

    Copyright (c) 2017-2023, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
import logging
import OurStatsModels       as sm
from ScatteringBaseUtil     import SMALL_ANGLE_LIMIT
from molass_legacy.Baseline.ScatteringBaseline     import ScatteringBaseline
from ScatteringBasecurve    import ScatteringBasecurve
from ScatteringBasespline   import ScatteringBasespline
# from molass_legacy.KekLib.NumpyUtils             import np_savetxt
from molass_legacy._MOLASS.SerialSettings         import get_setting
from molass_legacy.SerialAnalyzer.ElutionBaseCurve       import ElutionBaseCurve
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker

DEBUG_PLOT  = False

NUM_SLOPE_ESTIMATION    = 10
NUM_SAMPLE_CURVES       = 20
VERY_SMALL_ANGLE_LIMIT  = 0.03

def is_to_plot( self, i, q, very_small_angle_only=False ):
    # return q >= 0.01 and ( i == self.index or i % 8 == 0 )
    if very_small_angle_only:
        return q < VERY_SMALL_ANGLE_LIMIT
    else:
        return i == self.index or i % 8 == 0

class ScatteringBasesurface:
    def __init__( self, qvector, index, data,
                    inty_curve_y=None, parent=None, baseline_degree=None ):
        self.qvector    = qvector
        self.index      = index
        self.data       = data
        self.parent     = parent
        self.logger     = logging.getLogger( __name__ )

        self.size = size = self.data.shape[0]

        if baseline_degree is None:
            baseline_degree = get_setting( 'baseline_degree' )

        self.baseline_degree = baseline_degree

        # TODO: pass size_sigma instead of inty_curve_y
        #       See also apply_baseline_correction_impl
        if inty_curve_y is None:
            size_sigma = None
        else:
            ecurve = ElutionBaseCurve( inty_curve_y )
            size_sigma = ecurve.compute_size_sigma()
            print( 'default size_sigma=', size_sigma )

        self.size_sigma = size_sigma

        self.Y = np.linspace( 0, size-1, NUM_SAMPLE_CURVES, dtype=int )

        iv, line_array = self.compute_lines()

        self.iv = iv
        self.line_array = line_array

        self.iv_vsa = None
        self.line_array_vsa = None
        self.average_curve = None
        # self.compute_slope_param_fg()
        # self.compute_average_curve()

    def compute_lines( self, very_small_angle_only=False ):
        j = np.arange( self.data.shape[0] )
        i_list = []
        line_list = []
        continue_plot = True
        for i, q in enumerate( self.qvector ):
            if not is_to_plot( self, i, q, very_small_angle_only=very_small_angle_only ):
                continue

            i_list.append(i)
            Z = self.data[:,i,1]

            ecurve = ElutionBaseCurve( Z )

            if q < SMALL_ANGLE_LIMIT:
                p_final = ecurve.compute_base_percentile_offset( q, return_simple=True )
            else:
                p_final = ecurve.compute_base_percentile_offset( q, size_sigma=self.size_sigma )

            iterate = 2 if q < SMALL_ANGLE_LIMIT else 1
            baseline = np.zeros( len(Z) )
            try:
                for k in range(iterate):
                    if self.baseline_degree == 1:
                        sbl = ScatteringBaseline( Z - baseline )
                        A, B = sbl.solve( p_final=p_final )
                        z = A*j + B
                    elif self.baseline_degree == 2:
                        sbl = ScatteringBasecurve( Z - baseline )
                        F, D, E = sbl.solve( p_final=p_final )
                        z = F*j**2 + D*j + E
                    else:
                        sbl = ScatteringBasespline( Z - baseline, q=q )
                        sbl.solve( p_final=p_final )
                        z = sbl.get_baseline( j )

                    baseline += z
            except:
                etb = ExceptionTracebacker()
                self.logger.warning( str(etb) + ' with i=' + str(i) + ', k=' + str(k) + ', Z=' + str(Z) )
                # occured in Yonezawa-san data. See the 20170831 e-mail
                # self.logger.warning( etb )

            line_list.append( baseline[self.Y] )
            if DEBUG_PLOT and continue_plot:
                continue_plot = sbl.debug_plot( title='Debug at Q[%d] = %.3g' % (i, q), parent=self.parent )

        return np.array( i_list ), np.array( line_list )

    def compute_slope_param_fg( self ):
        # np_savetxt( 'line_array.csv', self.line_array )
        x = []
        y = []
        for i in self.iv[:NUM_SLOPE_ESTIMATION]:
            print( 'i=', i )
            for k, j in enumerate( self.Y ):
                x.append( j )
                y.append( self.line_array[i,k]/self.line_array[i,0] - 1 )

        if True:
            from DebugCanvas    import DebugCanvas
            def xy_plot( fig ):
                ax = fig.add_subplot( 111 )
                for i_ in range(NUM_SLOPE_ESTIMATION):
                    start = i_ * len( self.Y )
                    slice_ = slice( start, start+len( self.Y ) )
                    ax.plot( x[slice_], y[slice_], 'o', label=str(i_) )

                ax.legend()
                fig.tight_layout()

            dc = DebugCanvas( "Debug x, y", xy_plot, toolbar=True )
            dc.show()

        model = sm.OLS(y, x)
        results = model.fit()
        print( results.params )
        self.fg = results.params[0]
        return self.fg

    def compute_average_curve( self ):
        normalized_curves_list = []
        for k, j in enumerate( self.Y ):
           curve = self.line_array[ self.iv, k ] / ( self.fg * j + 1 )
           normalized_curves_list.append( curve )

        normalized_curves = np.array( normalized_curves_list )
        print( 'normalized_curves.shape=', normalized_curves.shape )
        self.average_curve = np.average( normalized_curves, axis=0 )
        print( 'self.average_curve.shape=', self.average_curve.shape )

    def plot( self, ax1, ax2=None, very_small_angle_only=False ):
        if ax2 is not None:
            ax2.set_title( "Background curves made with low percentile method" )

        if very_small_angle_only:
            iv = np.where( self.qvector < VERY_SMALL_ANGLE_LIMIT )[0]
            if self.line_array_vsa is None:
                iv, line_array = self.compute_lines( very_small_angle_only=True )
                self.iv_vsa = iv
                self.line_array_vsa = line_array
            iv = self.iv_vsa
            line_array = self.line_array_vsa
        else:
            iv = self.iv
            line_array = self.line_array

        i = 0
        for k, q in enumerate( self.qvector ):
            if not is_to_plot( self, k, q, very_small_angle_only=very_small_angle_only ):
                continue

            X = np.ones( len(self.Y) ) * q
            ax1.plot( X, self.Y, line_array[i,:], color='red', alpha=0.2 )
            i += 1

        for k, j in enumerate( self.Y ):
            x = self.qvector[iv]
            y = np.ones( len(iv) ) * j
            z = line_array[:,k]
            ax1.plot( x, y, z, color='yellow', alpha=0.5 )

            if ax2 is not None:
                ax2.plot( x, z, color='yellow' )

                if self.average_curve is not None:
                    if k == len(self.Y)//2:
                        a_curve = self.average_curve * ( self.fg * j + 1 )
                        ax2.plot( x, a_curve, color='green' )
       