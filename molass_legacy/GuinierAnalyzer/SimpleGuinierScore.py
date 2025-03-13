# coding: utf-8
"""
    SimpleGuinierScore.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy        as np
from scipy          import stats

MIN_GUINIER_SIZE        =  8            # required for Matsumura2-173 with QUALITY_BOUNDARY_QRG==0.5
SUF_GUINIER_SIZE        = 60            # required for Matsumura2-179, 180, 211 when END_GUINIER_SIZE==20
FWD_GUINIER_SIZE        = 10
END_GUINIER_SIZE        = 20
LINEARITY_SCALE         = 5
RG_CONSISTENCY_SCALE    = 20
QUALITY_BOUNDARY_IWD    = 0.1
IWD_CONSISTENCY_LIMIT   = 0.3
QUALITY_BOUNDARY_LOW    = 0.2
QUALITY_BOUNDARY_MID    = 0.7
QUALITY_BOUNDARY_HIGH   = 0.9
CONSISTENCY_BOUNDARY    = 0.5
USE_FWD_CONSISTENCY     = True

if USE_FWD_CONSISTENCY:
    SCORING_WEIGHTS     = np.array( [ 0.4, 0.5, 0.06, 0.04 ] )
    SCORING_WEIGHTS_MH  = np.array( [ 0.6, 0.3, 0.06, 0.04 ] )
    SCORING_WEIGHTS_ML  = np.array( [ 0.7, 0.3, 0.0, 0.0 ] )        # Matsumura2-198
    SCORING_WEIGHTS_LQ  = np.array( [ 0.8, 0.2, 0.0, 0.0 ] )
else:
    SCORING_WEIGHTS = np.array( [ 0.5, 0.4, 0.1 ] )
NUM_SCORE_FACTORS       = len(SCORING_WEIGHTS)
ZERO_SCORE_VECTOR = np.zeros(NUM_SCORE_FACTORS)
LOGARITHMIC_SCORING     = True
USE_GUINIER_PLOT_LENGTH = False         # should be True for 20160227-extrapolated-asc-003

DEBUG_PLOT = False

def compute_rg( slope ):
    if slope < 0:
        rg = np.sqrt( -3*slope )
    else:
        rg = 0
    return rg

def evaluate_rg_consistency( rg1, rg2 ):
    return 0 if rg2 == 0 else np.exp( -abs( 1 - rg1/rg2 )*RG_CONSISTENCY_SCALE )

def compute_fwd_consistency( start, stop, x2, log_y ):
    if not USE_FWD_CONSISTENCY:
        return None

    from_ = max( 0, start - FWD_GUINIER_SIZE )
    to_ = max( from_ + FWD_GUINIER_SIZE, start )

    rgs = []
    for slice_ in [ slice( from_, to_), slice(stop, stop + FWD_GUINIER_SIZE) ]:
        x_ = x2[slice_]
        y_ = log_y[slice_]
        slope = stats.linregress( x_, y_ )[0]
        rgs.append( compute_rg( slope ) )
    # print( 'compute_end_consistency: rgs=', rgs )
    return evaluate_rg_consistency( rgs[0], rgs[1] )

def compute_end_consistency( start, stop, x2, log_y, debug=False ):
    end_size = min( (stop-start)//2, END_GUINIER_SIZE )
    slice_ = slice( stop - end_size, stop )
    rgs = []

    debug_ = DEBUG_PLOT or debug

    if debug_:
        debug_intervals = []

    for slice_ in [ slice(start, start+end_size), slice(stop-end_size, stop) ]:
        x_ = x2[slice_]
        y_ = log_y[slice_]
        slope = stats.linregress( x_, y_ )[0]
        rgs.append( compute_rg( slope ) )
        if debug_:
            debug_intervals.append( [ x_, y_ ] )
    # print( 'compute_end_consistency: rgs=', rgs )

    if debug_:
        from CanvasDialog   import CanvasDialog
        slice_ = slice( 0, stop )
        x   = x2[slice_]
        y   = log_y[slice_]
        print( 'rgs=', rgs )
        def func( fig ):
            ax = fig.add_subplot(111)
            ax.set_title( 'compute_end_consistency')
            ax.plot( x, y, 'o', markersize=3 )
            for x_, y_ in debug_intervals:
                ax.plot( x_, y_, 'o', markersize=3 )
            fig.tight_layout()
        dialog = CanvasDialog( "Debug" )
        dialog.show( func, figsize=(9,8), toolbar=True )

    return evaluate_rg_consistency( rgs[0], rgs[1] )

def get_linearity( score_array ):
    # return np.prod( [ np.power( score_array[k+1], w ) for k, w in enumerate( [0.5, 0.5]) ] )
    return np.sum( [ score_array[k+1] * w for k, w in enumerate( [0.5, 0.5]) ] )

def compute_expected_size( px, dx, rg ):
    max_q = 1.3/rg - px
    expected_size = max( MIN_GUINIER_SIZE+1, int( max_q/dx ) )
    # max_size = min( SUF_GUINIER_SIZE, expected_size )
    return expected_size

def compute_expected_length( px, rg ):
    max_q = 1.3/rg
    dx  = max_q**2 - px**2      # width  in Guinier plot scale
    dy  = dx * (rg**2) /3       # height in Guinier plot scale
    length = np.sqrt( dx**2 + dy**2 )
    return length

def evaluate_guinier_interval( bq, peak, dx, rg, length, r_value, end_consistency, fwd_consistency, spline_rg, return_vector=False, debug=False ):
    if USE_GUINIER_PLOT_LENGTH:
        max_length  = compute_expected_length( peak, rg )
    else:
        max_length  = compute_expected_size( peak, dx, rg )
    if debug:
        print( 'peak=', peak, 'dx=', dx, 'rg=', rg, 'length=', length, 'max_length=', max_length )

    if  bq < QUALITY_BOUNDARY_LOW:
        weights         = SCORING_WEIGHTS_LQ
    elif bq > QUALITY_BOUNDARY_HIGH:
            weights     = SCORING_WEIGHTS
    else:
        if bq > QUALITY_BOUNDARY_MID:
            # print( 'end_consistency=', end_consistency )
            if end_consistency < CONSISTENCY_BOUNDARY:
                # neccesary e.g. for 20160227
                weights = SCORING_WEIGHTS
            else:
                weights = SCORING_WEIGHTS_MH
        else:
            weights = SCORING_WEIGHTS_ML

    if spline_rg is None:
        ratio = min( 1, length/max_length )
        size_score = ratio**2
    else:
        size_score = evaluate_rg_consistency( rg, spline_rg )

    if debug:
        print( 'size_score=', size_score )

    linearity_score = np.exp( -(r_value + 1)*LINEARITY_SCALE )

    if USE_FWD_CONSISTENCY:
        score_array = np.array( [size_score, linearity_score, end_consistency, fwd_consistency ] )
    else:
        score_array = np.array( [size_score, linearity_score, end_consistency ] )

    score_vector = weights * score_array

    if LOGARITHMIC_SCORING:
        score = np.prod( [ np.power( score_array[k], w ) for k, w in enumerate( weights ) ] )
    else:
        score = np.sum( score_vector )
    if return_vector:
        return score, score_vector, score_array
    else:
        return score
