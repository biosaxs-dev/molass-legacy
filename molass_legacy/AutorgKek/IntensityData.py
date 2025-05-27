"""
    IntensityData.py

    Copyright (c) 2016-2023, SAXS Team, KEK-PF
"""
import numpy        as np
from scipy          import stats
from molass_legacy.KekLib.NumpyUtils     import get_valid_index, np_loadtxt_robust
from HeadAnomalies  import HeadAnomalies

BASIC_CHECK_LENGTH          = 10
SLOPE_INDEX                 = 0
R_VALUE_INDEX               = 2
P_VALUE_INDEX               = 3
ACCEPTABLE_P_VALUE_LIMIT    = 0.3       # 20160615/1PCAF_00173_sub.dat, subtract_SSC/IN_c380_00001_sub.dat
ACCEPTABLE_P_RANGE_SIZE     = 4         # subtract_SSC/IN_c380_00001_sub.dat [5, 10]
ACCEPTABLE_P_MEDIAN         = 0.05      # 0.0424 for 20160615/1PCAF_00176_sub.dat
TOO_NARROW_P_MEDIAN         = ACCEPTABLE_P_MEDIAN + 0.01

ACCEPTABLE_R_VALUE_LIMIT    = 0.1
ACCEPTABLE_R_MEAN           = -0.7
TOO_NARROW_R_MEAN           = ACCEPTABLE_R_MEAN + 0.01
SUFFICIENT_R_MEAN           = -0.99
SUFFICIENTLY_LOW_ERROR      = 0.003
ACCEPTABLE_ERROR_SCORE      = 0.1
ACCEPTABLE_BASIC_QUALITY    = 0.8

"""
    p50=0.028 for 'IN_c380_00010_sub.dat'
    any other ?
"""

DEBUG = False
ACCEPTABLE_POSITIVE_RATIO   = 0.1

class IntensityData:
    def __init__( self, source, add_smoother=False ):

        self.comments = None

        if type( source ) == str:
            array_, comment_lines = np_loadtxt_robust( source, usecols=(0, 1, 2) )
            if len( comment_lines ) > 0:
                self.set_comments( comment_lines )

        elif type( source ) == np.ndarray:
            array_ = np.array( source )
        elif type( source ) == IntensityData:
            array_ = np.array( source.array )
        else:
            assert( False )

        self.orig_array = array_
        # print( 'orig_array.shape=', self.orig_array.shape )

        self.add_smoother_flag = add_smoother

        self.initialize()

    def set_comments( self, comment_lines ):
        self.comments = ''.join( comment_lines )

    def initialize( self ):
        array_ = self.orig_array

        I_ = array_[ :, 1 ]
        E_ = array_[ :, 2 ]
        # I_valid = get_valid_index( I_ )
        I_positive = np.logical_and( I_ > 0, E_ > 0 )
        array_ = array_[ I_positive, : ]
        self.positive_where = np.where( I_positive )
        self.positive_index = np.ones( ( I_.shape[0], ), dtype=int ) * ( -1 )
        for i, p in enumerate( self.positive_where[0] ):
            self.positive_index[ p ] = i

        self.array = array_
        self.positive_ratio = self.compute_positive_ratio()

        self.Q  = self.array[ :, 0 ]
        self.I  = self.array[ :, 1 ]
        self.E  = self.array[ :, 2 ]
        self.size = self.array.shape[0]
        # print( 'self.size=', self.size )
        self.X  = self.Q**2
        self.Y  = np.log( self.I )
        W_ = 1/self.E**2
        self.W  = np.sqrt( W_ / np.sum( W_ ) )
        if DEBUG:
            print( self.Q[0:5] )
            print( self.I[0:5] )
            print( self.E[0:5] )

        try:
            self.observe_basic_condition()
            self.compute_basic_range()
        except Exception as exc:
            if DEBUG:
                raise exc
            else:
                self.basic_quality = 0
                return

        self.smoother = None
        if self.add_smoother_flag:
            if self.positive_ratio >= ACCEPTABLE_POSITIVE_RATIO:
                try:
                    self.add_smoother()
                except:
                    raise RuntimeError( 'positive_ratio=%g' % self.positive_ratio )

    def has_acceptable_quality( self ):
        return self.basic_quality > 0 and self.positive_ratio >= ACCEPTABLE_POSITIVE_RATIO

    def copy( self, start=0 ):
        array_ = self.orig_array[ start:, : ]
        return IntensityData( array_, add_smoother=self.add_smoother_flag )

    def get_usable_range( self ):
        return self.f1, self.t1

    def get_approximate_range( self ):
        f = max( 10,  min( 20, self.f1 ) )
        return f, self.size // 4

    def get_guinier_valid_xy( self ):
        return self.X, self.Y, self.E

    def compute_positive_ratio( self ):
        I = self.orig_array[ :, 1 ]
        N = I.shape[0]

        y05, y95 = np.percentile( I, [ 5, 95 ] )
        if y05 > 0:
            if DEBUG: print( 'y05, y95=', y05, y95 )
            dy = ( y95 - y05 )/( N - 1 )
            weights = np.arange( y95, y05 - dy/10, -dy )
            if DEBUG: print( 'N=', N, 'weights.shape=', weights.shape )
        else:
            weights = np.ones( ( N, ) )

        # print( 'weights.shape=', weights.shape, 'N=', N )
        assert( weights.shape[0] == N )

        weights /= np.sum( weights )
        ratio = np.sum( weights[ I > 0 ] )
        if DEBUG: print( 'positive_ratio=', ratio )
        return ratio

    def observe_basic_condition( self ):
        i = 0
        at_end = False
        eval_array = []
        while i * BASIC_CHECK_LENGTH < self.X.shape[0] and not at_end:
            start   = i * BASIC_CHECK_LENGTH
            end     = start + BASIC_CHECK_LENGTH
            if end + BASIC_CHECK_LENGTH > self.X.shape[0]:
                end = self.X.shape[0]
                at_end = True
            result  = self.evaluate_range( slice( start, end ) )
            eval_array.append( result )
            i += 1

        array_ = np.array( eval_array )

        if DEBUG:
            from molass_legacy.KekLib.NumpyUtils import np_savetxt
            np_savetxt( "observe_basic_condition.csv", array_ )

        self.slope_array        = array_[:,SLOPE_INDEX]
        self.r_array = r_array  = array_[:,R_VALUE_INDEX]
        self.p_array = p_array  = array_[:,P_VALUE_INDEX]
        # print( 'p_array=', p_array[0:10] )

        self.p_acceptable = p_array < ACCEPTABLE_P_VALUE_LIMIT
        self.p_ok = p_ok = np.where( self.p_acceptable )[0]
        # print( 'p_ok=', p_ok )
        # TODO: case p_ok is empty

        t10_ok = False
        for k in range( min( 4, len( p_ok ) ) ):
            f10 = p_ok[k]

            # pick the continuous ( simply connected ) part at head
            head_scale = np.array( range( f10, f10 + p_ok.shape[0]-k ) )
            p_ok_head = np.where( p_ok[k:] == head_scale )[0]
            # print( 'p_ok_head=', p_ok_head )
            t10 = p_ok[ p_ok_head[-1] ]

            if t10 > f10 + 1:
                # print( 't10=', t10 )
                if t10 > 3:
                    t10_ok = True
                    break
                else:
                    # as in case of subtract_SSC/IN_c380_00010_sub.dat
                    continue

        if not t10_ok:
            k = min( 1, p_ok.shape[0]-1 )
            # k == 0 seems to be inappropriate
            # as for aggregation_20160524/AIMdim01_00171_sub.dat
            f10 = p_ok[k]
            t10 = f10

        if DEBUG: print( 'f10, t10=', [ f10, t10 ] )

        self.p50 = np.percentile( p_array[f10:t10+1], 50 ) if t10 >= f10 + ACCEPTABLE_P_RANGE_SIZE else TOO_NARROW_P_MEDIAN
        if DEBUG: print( 'self.p50=', self.p50 )
        self.p_value_ok = self.p50 <= ACCEPTABLE_P_MEDIAN
        if DEBUG: print( 'p50=', self.p50, ', basically_ok=', self.p_value_ok )

        self.basic_condition =  max( 0, ( ACCEPTABLE_P_MEDIAN - self.p50 ) / ACCEPTABLE_P_MEDIAN )

        self.f10 = f10
        self.t10 = t10

    def evaluate_range( self, slice_ ):
        X_ = self.X[ slice_ ]
        Y_ = self.Y[ slice_ ]
        slope, intercept, r_value, p_value, std_err = stats.linregress( X_, Y_ )
        return [ slope, intercept, r_value, p_value, std_err ]

    def get_acceptable_XYW( self ):
        pass

    def compute_basic_range( self ):

        f10 = self.f10
        t10 = self.t10

        if f10 > 0:
            f1 = None
            for j in range( (f10-1)*BASIC_CHECK_LENGTH, f10*BASIC_CHECK_LENGTH ):
                # print( 'j=', j )
                result  = self.evaluate_range( slice( j, j+BASIC_CHECK_LENGTH ) )
                if result[1] < ACCEPTABLE_P_VALUE_LIMIT:
                    f1 = j
                    break
            if f1 is None:
                f1 = f10*BASIC_CHECK_LENGTH
        else:
            f1 = 0

        t1 = None
        t1_max = min( self.X.shape[0], (t10+1)*BASIC_CHECK_LENGTH )
        for j in range( t10*BASIC_CHECK_LENGTH, t1_max ):
            # print( 'j=', j )
            result  = self.evaluate_range( slice( j, j+BASIC_CHECK_LENGTH ) )
            if result[P_VALUE_INDEX] < ACCEPTABLE_P_VALUE_LIMIT:
                t1 = j
            else:
                break

        if t1 is None:
            t1 = t1_max - 1

        head_slice = slice( 0, self.size//4 )
        Ih_ = self.I[head_slice]
        Xh_ = self.X[head_slice]
        Yh_ = self.Y[head_slice]
        ha = HeadAnomalies( Ih_, Xh_, Yh_ )
        ha.remove_head_anomalies()
        if DEBUG: print( 'ha.start_point=', ha.start_point )
        self.ha_end = ha.start_point

        f_min = ha.start_point
        # f1 = max( f1, f_min )
        f1  = f_min

        if DEBUG: print( 'p_value_ok=', self.p_value_ok )
        if self.p_value_ok:
            f_ = f1
            t_ = max( f1 + 100, t1 )
        else:
            f_ = 10
            t_ = f_ + 200

        self.basic_quality_reg = e = self.evaluate_range( slice( f_, t_+1 ) )
        self.basic_error = abs( e[-1]/e[0] )
        if DEBUG:
            print( 'f_, t_=', [ f_, t_ ] )
            print( 'slope=', e[0], ', std_err=', e[-1], ', basic=', self.basic_error )
        self.basic_quality = np.exp( - max( 0, self.basic_error - SUFFICIENTLY_LOW_ERROR )*10 )
        self.basically_ok = self.basic_quality >= ACCEPTABLE_BASIC_QUALITY

        if DEBUG: print( 'f_, t_=', [ f_, t_ ], ', basic_quality=', self.basic_quality )

        if self.basically_ok:
            f_ = f1
            t_ = max( f1 + 100, t1 )
        else:
            f_ = 10
            t_ = f_ + 200

        self.f1 = f_
        self.t1 = t_

    def add_smoother( self ):
        from GaussianProcessDep     import SmootherChain

        if DEBUG: print( 'add_smoother: basic_quality=', self.basic_quality )
        if self.basically_ok:
            # f1, t1 = self.f1, self.t1
            f1, t1 = self.get_usable_range()
            f1 = min( 30, f1 )
            if t1 - f1 < 40:
                # f1, t1 = self.get_approximate_range()
                t1 = f1 + 40
        else:
            f1, t1 = self.get_approximate_range()

        if DEBUG: print( 'add_smoother: f1, t1=', f1, t1 )

        ft_size = t1 - f1
        quarter = ft_size // 4
        half    = quarter * 2

        boundaries  = [ f1, f1 + quarter, f1 + half, t1, self.size ]
        # print( 'boundaries=', boundaries )
        if self.basically_ok:
            nuggets = [ 1e-3, 1e-3, 1e-3, 1e-3 ]
        elif self.basic_quality >= 0.3:
            nuggets = [ 1e0, 1e0, 1e0, 1e0 ]
        else:
            nuggets = [ 1e1, 1e1, 1e1, 1e1 ]

        self.smoother = SmootherChain( self.Q, self.I, boundaries=boundaries, nuggets=nuggets )
        # self.smoother = Smoother( self.Q, self.I )

        # smooth data for aggregation check
        if self.basically_ok:
            self.slice_ = slice_ = slice( f1, t1+1 )
            self.SQ = self.Q[ slice_ ]
            self.SX = self.X[ slice_ ]
            W_ = self.W[ slice_ ]**2
        else:
            self.SQ = self.Q
            self.SX = self.X
            W_ = self.W**2
        self.SI = self.smoother( self.SQ )
        self.SY = np.log( self.SI )
        self.SW = np.sqrt( W_ / np.sum( W_ ) )
