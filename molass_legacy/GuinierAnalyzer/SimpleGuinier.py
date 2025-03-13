"""
    SimpleGuinier.py

    Copyright (c) 2017-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy import stats
import logging
from bisect import bisect_right
from molass_legacy.KekLib.NumpyUtils import np_loadtxt_robust
from molass_legacy.KekLib.OurCurvature import curvature_curve
from SimpleGuinierScore import ( MIN_GUINIER_SIZE, ZERO_SCORE_VECTOR, END_GUINIER_SIZE,
                                    QUALITY_BOUNDARY_HIGH, USE_GUINIER_PLOT_LENGTH,
                                    compute_rg,
                                    compute_expected_size,
                                    evaluate_rg_consistency,
                                    evaluate_guinier_interval,
                                    compute_end_consistency,
                                    compute_fwd_consistency )
from molass_legacy._MOLASS.SerialSettings import get_setting

NUM_INITIAL_IN_KNOTS    = 2
KNOT_NO_ADDITION        = 0
KNOT_DROPPING_HEAD      = 1
KNOT_SOARING_HEAD       = 2
MIN_NUM_HEAD_DROPS      = 10
FWD_NUM_HEAD_DROPS      = 20            # required for 20170301-207
TODO_DROPPING_RATIO     = 1.5
TODO_SOARING_RATIO      = 0.3           # required < 0.54 for SUB_TRN1PY-189, < 0.40 for TSsonme2-224
GOOD_SOARING_RATIO      = 1.8           # required > 1.64 for SUB_TRN1PY-195, < 1.98 for 20160227-0
ADEDUITE_R_VALUE        = -0.998
CANDIDATE_R_VALUE       = -0.96
ALTERNATIVE_R_VALUE     = -0.95
MONOTONICITY_SCALE      = 0.5
QUALITY_BOUNDARY_WORST  = 0.005         # 0.0073 for Matsumura2-172
QUALITY_BOUNDARY_LOW    = 0.02          # for SUB_TRN1_2-206, < 0.022 for 20161202-168
QUALITY_BOUNDARY_QRG    = 0.5           # 
QUALITY_BOUNDARY_BEST   = 0.95
ALLOW_FWD_CONSISTENCY   = 0.5
ALLOW_FWD_CONSISTENCY2  = 0.1
BACK_R_VALUE_PENALTY    = 0.3
MAX_NUM_GOING_BACK      = 40
PEAK_FIND_ROTATION      = -np.pi/6      # -π/6, i.e. 30°clockwise
PEAK_FIND_R_VALUE       = -0.03         # required for AhRR-10
PEAK_FIND_R_VALUE_ALLOW = -0.05
PEAK_FIND_TOO_LONG      = 40            # < 44 for SUB_TRN1-165, < 70 for 3/9-196
PEAK_FIND_RETRIABLE_LEN = 25
PEAK_FIND_PERCENTILE    =  5
PEAK_FIND_QUALITY_BNDRY = 50            # 58.5 for AhRR-12
PEAK_FIND_MIN_RG_RATIO  = 3
PEAK_FIND_RG_RATIO_LOW  = 2             # < 2.16 for TRN1PY-195, > 1.38 for Matsumura-258,  > 1.06 for 12/2-113
PEAK_FIND_MIN_AR_RATIO  = 0.2           # > 0.18 for Open01-232
PEAK_FIND_QRG_LIMIT     = 1.5           # > 
PEAK_FIND_MIN_CURVATURE = 0.0003        # < 0.00034 for SUB_TRNPY-193, > 0.00018 for Close01-205
MONOTONICITY_FWD_OK     = 0.5           # > 0.30 for Matsumura-258
TANGENT_DIFF_BOUNDARY   = 0.006         # > 0.005 for Open01-257
GOOD_MONOTONICITY_RATIO = 3             # > 2.15 for AhRR-10
TRY_PEAK_MIN_OFFSET     = 3
UP_ADJUST_X_DEV_BNDRY   = 0.01          # > 0.0032 for SUB_TRN1-160, < 0.017 for Matsumura2-172
LINREG_INDEX            = slice(0,3,2)
LINREG_STDERR           = 4
MIN_BETTER_RG_RATIO     = 75            # > 69 for 20160227-0,  < 81 for AHRR-12,  < 131 for AhRR-133
MAX_START_Q2            = 0.0005        # 
MIN_START_FROM_PEAK     = 5             # required for 20160628/AIMdim01_00179_sub.dat
SMALL_Q2_BOUNDARY       = MAX_START_Q2/2
SCORE_RATIO_BOUNDARY    = 0.7
WST_AT_LEAST_TRIALS     = 2
LQ_AT_LEAST_TRIALS      = 3

EXCEPTION_LOG   = True
DEBUG_PLOT      = False

def rotate( th, wx, wy ):
    cx  = wx[-1]
    cy  = wy[-1]
    wx_ = wx - cx
    wy_ = wy - cy
    c   = np.cos( th )
    s   = np.sin( th )
    return cx + ( wx_*c - wy_* s ), cy + ( wx_*s + wy_* c )

def compute_monotonicity( y ):
    height = y[0] - y[-1]

    if height > 0:
        gry     = np.gradient( y )
        ag_sum  = np.sum( np.abs( gry ) )
        ag_ratio = ag_sum / height
        monotonicity = np.exp( MONOTONICITY_SCALE *( 1 - ag_ratio ) )
        avg_gradient = np.average( gry )
    else:
        monotonicity = 0
        avg_gradient = 0

    return monotonicity, avg_gradient

class SimpleGuinier:
    def __init__( self, file, curvature=False, basic_quality_only=False, anim_data=False, debug_plot=False ):
        self.logger = logging.getLogger( __name__ )
        self.debug_plot = debug_plot or DEBUG_PLOT
        if type( file ) == str:
            self.data, _ = np_loadtxt_robust( file )
        elif type( file ) == np.ndarray:
            self.data = file
        else:
            raise RuntimeError()

        # print( self.data.shape )
        self.curvature  = curvature
        self.anim_data  = anim_data
        if self.anim_data:
            self.best_history_dict = {}
        self.x_ = x_ = self.data[ :, 0 ]
        self.y_ = y_ = self.data[ :, 1 ]
        self.e_ = e_ = self.data[ :, 2 ]
        positive = np.logical_and( y_ > 0, e_ > 0 )

        self.qrg_stop = None
        self.rg_ratio_for_better_peak = None
        self.head_gap_ratio = None
        self.basic_quality = None
        self.too_long_peak_candidate = None
        self.anim_guinier_stop = None
        self.peak_type  = None
        self.try_ret    = None
        self.tester_log_writer = None

        try:
            self.x = x_[positive]
            self.y = y_[positive]
            self.e = e_[positive]
            self.dx = x_[1] - x_[0]
            self.x2 = self.x**2
            self.log_y = np.log(self.y)
            self.log_y_err = 1/self.y * self.e  # error propagation of logarithm: error log(y) = 1/y * y_error
            if len( self.log_y ) > MIN_GUINIER_SIZE * 4:
                self.make_smooth_line()
                self.evaluate_basic_quality()
            else:
                # as in 20170426-21
                self.logger.warning("SimpleGuinier aborted due to %d <= %d", len( self.log_y ), MIN_GUINIER_SIZE * 4)
        except:
            if EXCEPTION_LOG:
                from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
                etb = ExceptionTracebacker()
                self.logger.error( etb )
            else:
                pass

        if basic_quality_only:
            return

        if self.basic_quality is None:
            self.set_guinier_null_result()
        else:
            self.guinier_interval()

    def make_smooth_line(self, debug=False):
        n = len(self.x)
        points = [ n ]
        for i in range(NUM_INITIAL_IN_KNOTS):
            n //= 2
            points.append(n)
        internal_points = np.array( list( reversed( points[1:] ) ), dtype=int )
        knots = self.x2[internal_points]

        self.added_knot_type = KNOT_NO_ADDITION
        self.soaring_peaks  = 0
        smoother = LSQUnivariateSpline(self.x2, self.log_y, knots)
        head_slice = slice(0,internal_points[0])
        head_x = self.x2[head_slice]
        head_y = self.log_y[head_slice] 
        head_len = len(head_y)

        # select 98-percentille point as peak-point to avoid abnormal max
        kth = int( head_len*98/100 )
        arg_hy98 = np.argpartition( head_y, kth )
        m = arg_hy98[kth]
        if self.debug_plot:
            print( 'head_len=', head_len, 'kth=', kth, 'm=', m )
        if m < MIN_NUM_HEAD_DROPS:
            gr_fwd = np.gradient( head_y[m:m+FWD_NUM_HEAD_DROPS] )
            gr_ratio = len( np.where( gr_fwd > 0 )[0] ) / FWD_NUM_HEAD_DROPS
            if self.debug_plot:
                print( 'gr_ratio=', gr_ratio, 'gr_fwd=', gr_fwd )
            # if it has a significant increasing part, try again for the next forward part
            if gr_ratio > 0.35:
                kth = int( ( head_len - m )*98/100 )
                arg_hy98 = np.argpartition( head_y[m:], kth )
                m = m + arg_hy98[kth]
                if self.debug_plot:
                    print( 'kth=', kth, 'm=', m )

        # if self.debug_plot:
        if False:
            from CanvasDialog   import CanvasDialog
            def func( fig ):
                ax = fig.add_subplot(111)
                ax.set_title( 'Peak Debug')
                ax.plot( head_x, head_y, 'o', markersize=3 )
                ax.plot( head_x[m], head_y[m], 'o', color='red', markersize=3  )
                ax.plot( head_x[0:m], head_y[0:m], 'o', color='orange', markersize=3  )
                ax.plot( head_x, smoother(head_x), color='green' )
                ax.set_xlim( ax.get_xlim() )
                ax.set_ylim( ax.get_ylim() )
                ax.plot( knots, smoother(knots), 'o', color='red', markersize=3)
                fig.tight_layout()
            dialog = CanvasDialog( "Debug" )
            dialog.show( func, figsize=(9,8), toolbar=True )

        self.stop = internal_points[0]  # note that this setting is not final

        self.fixed_guinier_start = get_setting( 'fixed_guinier_start' )
        if self.fixed_guinier_start == 1:
            m = get_setting( 'guinier_start_point' )
            self.guinier_start_point = m
            self.set_peak_info( m, smoother )

        else:

            self.set_peak_info( m, smoother )

            if m > 1:
                # pre_head_mean = np.average( head_y[0:m] )
                pre_head_mean   = np.percentile( head_y[0:m], 50 )
            else:
                pre_head_mean   = head_y[m]
            # print( 'pre_head_mean=', pre_head_mean, 'log_py=', self.log_py, 'head_y[0:m]=', head_y[0:m] )

            if m >= MIN_NUM_HEAD_DROPS and pre_head_mean < self.log_py:
                height_base = smoother(self.x2[0])
                height_p    = self.log_py  - height_base
                height_sp   = self.log_spy - height_base
                h_ratio = height_p/height_sp
                if height_sp > 0 and h_ratio >= TODO_DROPPING_RATIO:
                    self.peak_type  = '1-1'
                    # this indicates dropping head
                    knots = np.array( [self.px2] + list( knots ) )
                    self.added_knot_type = KNOT_DROPPING_HEAD
                    # print( 'knots=', knots )
                    smoother = LSQUnivariateSpline(self.x2, self.log_y, knots)
                    # why don't do set_peak_info?
                else:
                    self.peak_type  = '1-0'
            else:
                log_ym      = self.log_y[m]
                log_sym     = smoother( self.x2[m] )
                head_gap1   = log_ym - log_sym
                # head_gap2   = log_ym - smoother( self.x2[internal_points[0]] )
                if head_gap1 > 0:
                    self.peak_type  = '0-1'
                    m20 = min( len( self.x2 ) - 1, 20 )     # len( self.x2 )==15 for 20160426-9
                    head_gap2   = log_sym - smoother( self.x2[m20] )
                    # print( 'head_gap1=', head_gap1, 'head_gap2=', head_gap2 )
                    if head_gap2 > 0:
                        head_gap_ratio = head_gap1/head_gap2
                    else:
                        head_gap_ratio = TODO_SOARING_RATIO + 10
                        # i.e., always do find_better_peak

                    # print( 'head_gap_ratio=', head_gap_ratio )
                    self.head_gap_ratio = head_gap_ratio
                    if head_gap_ratio > TODO_SOARING_RATIO:
                        self.peak_type  = '0-1-1'
                        # case like pH6.136
                        m_try = self.find_better_peak( head_gap_ratio, m, head_x[m:], head_y[m:] )
                        if m_try is not None:
                            self.peak_type  = '0-1-1-1'
                            for t_ in range(2):
                                ret_smoother = self.try_better_peak( head_gap_ratio, m, m_try, knots, smoother, head_x, head_y )
                                if ret_smoother is not None:
                                    self.peak_type  = '0-1-1-1-1'
                                    smoother = ret_smoother
                                    self.added_knot_type = KNOT_SOARING_HEAD
                                    self.set_peak_info( m_try, smoother )
                                    break

                                if m_try < PEAK_FIND_RETRIABLE_LEN:
                                    break
                                else:
                                    # as in TSsome2-225
                                    m_try = self.find_better_peak( head_gap_ratio, m, head_x[m:m_try], head_y[m:m_try] )
                                    if m_try is None:
                                        break
                                if debug:
                                    print('retrying: for m_try=', m_try)
                else:
                    self.peak_type  = '0-0'

        self.log_sy = smoother(self.x2)
        self.log_sy_max = np.max( self.log_sy[0:internal_points[0]] )
        self.kx2 = smoother.get_knots()
        self.log_ky = smoother(self.kx2)
        self.sy = np.exp(self.log_sy)
        self.kx = np.sqrt(self.kx2)
        self.ky = np.exp(self.log_ky)
        self.stop = internal_points[0]

    def set_peak_info( self, m, smoother ):
        self.peak = m
        self.px  = self.x_[m]
        self.px2 = self.x2[m]
        self.log_py = self.log_y[m]
        self.log_spy = smoother( self.px2 )

    def find_better_peak(self, head_gap_ratio, start, head_x, head_y, debug=False):
        """
        note that we can't yet use self.basic_quality here
        because evaluate_basic_quality depends the peak info determined here
        """
        # scale x so that the slope be equal to -1 and rotate -pi/9
        # changed -pi/4 to -pi/6 for AhRR-124
        span_y = head_y[-1] - head_y[0]
        span_x = head_x[-1] - head_x[0]
        span_ratio = abs( span_y/span_x )
        hx = head_x * span_ratio

        # if self.debug_plot:
        if False:
            from CanvasDialog   import CanvasDialog
            def func( fig ):
                ax = fig.add_subplot(111)
                ax.set_title( 'hx, head_y')
                ax.plot( hx, head_y )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug" )
            dialog.show( func, figsize=(9,8), toolbar=True )

        # consider changing angle to -np.pi/4 for Matsumura2-78
        # print( 'span_ratio=', span_ratio )
        angle = PEAK_FIND_ROTATION
        x, y = rotate( angle, hx, head_y )

        def get_partition_index( percentile ):
            kth = int( len(x)*percentile/100 )
            x_argp = np.argpartition( x, kth )
            return x_argp[0:kth+1]

        p_index = get_partition_index( PEAK_FIND_PERCENTILE )
        # print( 'p_index=', p_index )

        x_  = x[p_index]
        y_  = y[p_index]

        # center of gravity
        cx  = np.average( x_ )
        cy  = np.average( y_ )

        pos_type = None

        # for TRN1PY-195, head_gap_ratio==1.64, 'largest' should be avoided
        if head_gap_ratio > GOOD_SOARING_RATIO:
            x_width = np.max( x - cx )
            x_dev = np.std( x_ ) / x_width
            # print( 'x_dev=', x_dev, 'x_width=', x_width )
            if x_dev < UP_ADJUST_X_DEV_BNDRY:
                # in cases such as SUB_TRN1-160
                # largest y_
                y_  = y[p_index]
                k   = np.argmax( y_ )
                m   = p_index[k]
                pos_type = 'largest'

        if pos_type is None:
            # in cases such as pH7-243
            # find nearest to the center of gravity
            d2  = (x_ - cx)**2 + (y_ - cy)**2
            k   = np.argmin( d2 )
            m   = p_index[k]
            pos_type = 'nearest to center'

        # print( 'find_better_peak: m=', m )

        """
        span_rx = np.max( np.abs( x - x[m] ) )
        span_ry = np.max( np.abs( y - y[-1] ) )
        span_ratio = span_ry / span_rx
        print( 'span_ratio=', span_ratio )
        """

        if self.debug_plot:
            from molass_legacy.KekLib.CanvasDialog import CanvasDialog
            def func( fig ):
                ax = fig.add_subplot(111)
                ax.set_title( '%g° rotated x, y' % ( 180*angle/np.pi ) )
                ax.plot( x[0:m], y[0:m], 'o', markersize=3 )
                tx, ty = x[m], y[m]
                ax.plot( x[m:], y[m:], 'o', markersize=3 )
                for k in p_index:
                    ax.plot( x[k], y[k], 'o', color='orange', markersize=3 )
                ax.plot( cx, cy, 'o', color='yellow' )
                ax.plot( tx, ty, 'o', color='red' )
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                xoffset = (xmax - xmin)*0.1
                yoffset = (ymax - ymin)*0.2
                if tx < xmin + xoffset:
                    xoffset *= -1
                    yoffset *= -0.5
                text = 'center of gravity for %d or lower percentile points' % PEAK_FIND_PERCENTILE
                ax.annotate( text, xy=( cx, cy ),
                    xytext=( cx - xoffset, cy + yoffset ), alpha=0.5,
                    arrowprops=dict( headwidth=5, width=1, facecolor='black', shrink=0, alpha=0.3),
                    # ha='center',
                    )
                text = pos_type + ' among %d or lower percentile points' % PEAK_FIND_PERCENTILE
                ax.annotate( text, xy=( tx, ty ),
                    xytext=( tx - xoffset, ty - yoffset ), alpha=0.5,
                    arrowprops=dict( headwidth=5, width=1, facecolor='black', shrink=0, alpha=0.3),
                    # ha='center',
                    )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug" )
            dialog.show( func, figsize=(9,8), toolbar=True )

        try:
            slope, r_value = stats.linregress( x[m:], y[m:] )[LINREG_INDEX]
            # print( 'find_better_peak: (1) r_value=', r_value )
            min_r_value = None
            min_r_n     = None

            if r_value > PEAK_FIND_R_VALUE:
                # try modifying m to get an expected situation
                m_stop = max( start + 5, m - MAX_NUM_GOING_BACK )
                for n in range( m-1,  m_stop, -1):
                    slope, r_value = stats.linregress( x[n:], y[n:] )[LINREG_INDEX]
                    if self.debug_plot:
                        print( 'find_better_peak: (2)', [n], 'r_value=', r_value )
                    if r_value <= PEAK_FIND_R_VALUE:
                        if debug:
                            print( 'modified m=%d to %d' % ( m, n ) )
                        m = n
                        break
                    if min_r_value is None or r_value < min_r_value:
                        min_r_value = r_value
                        min_r_n     = n

            # print( 'find_better_peak: r_value=', r_value, 'min_r_value=', min_r_value )
            if ( r_value > PEAK_FIND_R_VALUE
                and min_r_value is not None and min_r_value > PEAK_FIND_R_VALUE_ALLOW ):
                # not appropriate because the linearity is not sufficient
                m = None
            else:
                bck_slope, bck_r_value, bck_stderr = self.compute_backward_r_value( r_value, m, head_x, head_y )
                rg = compute_rg( slope )
                bck_rg = compute_rg( bck_slope )
                rg_ratio = bck_rg / rg if rg > 0 else 1e10
                bck_error = bck_stderr/abs( head_y[0] - head_y[-1])
                # print( 'bck_r_value=', bck_r_value, 'rg_ratio=', rg_ratio, 'bck_error=', bck_error )
                self.rg_ratio_for_better_peak   = rg_ratio
                # use bck_stderr because we can't use self.basic_quality yet
                # checking bck_error should avoid misjudgement for 20160227-23, etc.
                if bck_slope == 0:  # for Matsumura2-78
                    pass
                elif bck_error > PEAK_FIND_QUALITY_BNDRY and rg_ratio > MIN_BETTER_RG_RATIO:
                    # this peak adoption is valid onlyfor
                    #       for bad-quality data
                    #   and with big rg_ratio
                    # e.g. in case AhRR-133
                    pass
                else:
                    if head_gap_ratio > GOOD_SOARING_RATIO:
                        if ( bck_r_value < CANDIDATE_R_VALUE        # neccesary e.g. for 20160227
                            or bck_r_value + BACK_R_VALUE_PENALTY < r_value ):
                            # do not adopt this peak because the backward section has better linearity
                            m = None
        except:
            if True:
                from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
                etb = ExceptionTracebacker()
                print( etb )
            m = None

        if m is not None:
            self.soaring_peaks  += 1
            if m > PEAK_FIND_TOO_LONG:
                # TODO: better decision
                # save this peak candidate for stop point in make_cadidate_pairs
                # i.e. it would be inappropriate if we exceed this point
                # examples for this case include 20160227-124
                self.too_long_peak_candidate = start + m
                if debug:
                    print( 'too_long_peak_candidate=',  self.too_long_peak_candidate)
                m = self.find_better_peak( head_gap_ratio, start, head_x[0:m], head_y[0:m] )
                if m is not None:
                    m += start
            else:
                pass

        # print( 'find_better_peak: finally m=', m )

        return m

    def try_better_peak(self, head_gap_ratio, m, m_try, knots, smoother, head_x, head_y, debug=False):
        """
        we can adopt this peak if it substantially changes the spline
        """
        px = self.x2[m_try]
        # print( 'try_better_peak: ', knots )

        if ( m_try - m < TRY_PEAK_MIN_OFFSET    # like Jun28-157
            or px > knots[0]                    # Interior knots t must satisfy Schoenberg-Whitney conditions
            ):
            return None

        knots = np.array( [px] + list( knots ) )
        # print( 'knots=', knots )
        new_smoother = LSQUnivariateSpline(self.x2, self.log_y, knots)

        hx  = head_x[m]
        # bm  = min( m+MIN_GUINIER_SIZE, m_try )
        bm  = m + MIN_GUINIER_SIZE
        bslice = slice(m, bm)
        bx_ = self.x2[bslice]
        by_ = self.log_y[bslice]
        slope_bck, _ = stats.linregress( bx_, by_ )[LINREG_INDEX]

        px  = self.x2[m_try]
        fx  = self.x2[m_try+MIN_GUINIER_SIZE]
        shy, spy, sfy = new_smoother( [hx, px, fx] )
        slope_fwd   = ( sfy - spy ) / ( fx - px )

        rg_bck      = compute_rg( slope_bck )
        rg_fwd      = compute_rg( slope_fwd )
        if rg_fwd > 0:
            rg_ratio    = rg_bck / rg_fwd
        else:
            rg_ratio    = PEAK_FIND_MIN_RG_RATIO * 10

        # compute both( bck and fwd ) monotonicity
        # for judging from only one of them is risky
        bck_monotonicity, bck_tangent = compute_monotonicity( head_y[m:m_try] )
        fwd_monotonicity, fwd_tangent = compute_monotonicity( head_y[m_try:] )
        monotonicity_ratio = bck_monotonicity / fwd_monotonicity if fwd_monotonicity > 0 else 0

        # compute the tangent of the angle difference
        tangent_diff = ( fwd_tangent - bck_tangent ) / ( 1 + fwd_tangent * bck_tangent )

        peak_fwd_qrg    = self.x[m_try+MIN_GUINIER_SIZE] * rg_fwd

        if self.debug_plot:
            print( 'm=', m, 'm_try=', m_try )
            print( 'rg_ratio=', rg_ratio )
            print( 'bck_monotonicity=', bck_monotonicity )
            print( 'fwd_monotonicity=', fwd_monotonicity )
            print( 'monotonicity_ratio=', monotonicity_ratio )
            print( 'bck_tangent=', bck_tangent )
            print( 'fwd_tangent=', fwd_tangent )
            print( 'tangent_diff=', tangent_diff )
            print( 'peak fwd qrg=', peak_fwd_qrg )

            from molass_legacy.KekLib.CanvasDialog   import CanvasDialog
            hy  = head_y[0]
            py  = self.log_y[m_try]

            def func( fig ):
                ax = fig.add_subplot(111)
                ax.set_title( 'try_better_peak')
                ax.plot( head_x, head_y, 'o', markersize=3 )
                ax.plot( head_x, smoother(head_x) )
                ax.plot( head_x, new_smoother(head_x) )
                ax.plot( hx, hy, 'o', color='red' )
                ax.plot( hx, smoother(hx), 'o', color='yellow' )
                ax.plot( hx, new_smoother(hx), 'o', color='pink' )
                ax.plot( px, py, 'o', color='orange' )
                ax.plot( px, new_smoother(px), 'o', color='orange' )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug" )
            dialog.show( func, figsize=(9,8), toolbar=True )

        if peak_fwd_qrg > PEAK_FIND_QRG_LIMIT:
            # this case, as in Open01-232, is useless because no cadidates would be found
            self.try_ret = -1
            return None

        # the order of the following tests are relevant
        # so that you need to be very careful in maintenancer

        if fwd_monotonicity > MONOTONICITY_FWD_OK:
            if tangent_diff > TANGENT_DIFF_BOUNDARY:
                self.try_ret = 1
                return new_smoother

            cv = curvature_curve( head_x, head_y, spline=new_smoother )
            cv_try = cv[m_try]
            # print( 'cv_try=', cv_try )
            if cv_try > PEAK_FIND_MIN_CURVATURE:
                # as in SUB_TRNPY-193
                self.try_ret = 10
                return new_smoother
            else:
                self.try_ret = -10
                return None

        if fwd_monotonicity < QUALITY_BOUNDARY_LOW:
            if rg_ratio < PEAK_FIND_RG_RATIO_LOW:
                # like Matsumura-166, 12/2-81
                self.try_ret = -11
                return None

            if rg_ratio > PEAK_FIND_MIN_RG_RATIO:
                # like Nov24-209
                self.try_ret = 11
                return new_smoother

            # compute possible_size(=num points) at this point from the spline
            psy = new_smoother( px )
            fx  = self.x2[m_try + MIN_GUINIER_SIZE]
            fsy = new_smoother( fx )
            slope = ( fsy - psy )/( fx - px )
            rg = compute_rg( slope )
            max_q = 1.3/rg if rg > 0 else 1
            possible_size = max( 0, max_q - self.x[m_try] ) / self.dx
            if debug:
                print( 'possible_size=', possible_size, 'rg=', rg, 'max_q=', max_q, 'self.x[m_try]=', self.x[m_try], 'self.dx=', self.dx )
            if possible_size < MIN_GUINIER_SIZE:
                # like ?
                self.try_ret = -12
                return None
            else:
                # like AhRR-10, AhRR-134,
                # before checking ratio, too bad fwd_monotonicity should be avoided
                # note that this includes the case like HIF-74 where fwd_monotonicity==0
                self.try_ret = 12
                return new_smoother

        if monotonicity_ratio > GOOD_MONOTONICITY_RATIO:
            # there is a good possibility in backward part
            # so, adopting this peak is not a good idea
            self.try_ret = -2
            return None

        if ( fwd_monotonicity < QUALITY_BOUNDARY_QRG
            and rg_ratio < PEAK_FIND_RG_RATIO_LOW ):
            # for Matsumura-258
            self.try_ret = -3
            return None

        if fwd_monotonicity > QUALITY_BOUNDARY_QRG:
            self.try_ret = 2
            return new_smoother

        if fwd_monotonicity < QUALITY_BOUNDARY_LOW and rg_ratio < PEAK_FIND_RG_RATIO_LOW:
            # for 12/2-113
            self.try_ret = -4
            return None

        if head_gap_ratio <=  GOOD_SOARING_RATIO:
            # for test_SUB_TRN1PY-195
            self.try_ret = 3
            return new_smoother

        if  bck_monotonicity > QUALITY_BOUNDARY_BEST:
            # even if monotonicity_ratio is not enough
            # very good bck_monotonicity suggest a good possibility
            # assert False        # to see if this case be included monotonicity_ratio > GOOD_MONOTONICITY_RATIO
            # return None
            # for Ald-136, returning None is irrelevant
            pass

        if rg_ratio > PEAK_FIND_MIN_RG_RATIO:
            # too big rg ratio seems to indicate unacceptable soaring
            self.try_ret = 4
            return new_smoother

        self.try_ret = -5
        return None

    def compute_backward_r_value( self, r_value, m, head_x, head_y ):
        size = min( 40, len(head_x) - m )

        start = max( 0, m-size )
        slice_ = slice( start, m )

        if m - start < MIN_GUINIER_SIZE:
            return 0, 0, 1e5

        if False:
            from CanvasDialog   import CanvasDialog
            def func( fig ):
                ax = fig.add_subplot(111)
                ax.set_title( 'compute_backward_r_value')
                ax.plot( head_x, head_y, 'o' )
                ax.plot( head_x[slice_], head_y[slice_] )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug" )
            dialog.show( func, figsize=(9,8), toolbar=True )

        reg_ret = stats.linregress( head_x[slice_], head_y[slice_] )
        slope, r_value = reg_ret[LINREG_INDEX]
        stderr = reg_ret[LINREG_STDERR]
        return slope, r_value, stderr

    def evaluate_basic_quality(self, debug=False):
        log_y = self.log_y[self.peak:self.stop]

        if len(log_y) < MIN_GUINIER_SIZE:
            self.basic_quality = None
            if debug:
                self.logger.warning("basic_quality is set to None due to %d < %d", len(log_y), MIN_GUINIER_SIZE)
            return

        height = log_y[0] - log_y[-1]
        if height > 0:
            ag_sum = np.sum( np.abs( np.gradient( log_y ) ) )
            self.ag_ratio = ag_sum / height
            self.basic_quality = np.exp( MONOTONICITY_SCALE *( 1 - self.ag_ratio ) )
        else:
            self.basic_quality = 0

        self.worst_quality = self.basic_quality < QUALITY_BOUNDARY_WORST

    def make_cadidate_pairs( self, qrg_limit ):
        # print( 'make_cadidate_pairs: qrg_limit=', qrg_limit )

        if self.worst_quality:
            wide_allow  = 0.5
        elif self.basic_quality > QUALITY_BOUNDARY_QRG:     # 0.5
            wide_allow  = 0.3
        elif self.basic_quality < QUALITY_BOUNDARY_LOW:     # 0.02
            wide_allow  = 1.0
        else:
            wide_allow  = 0.5

        qrg_limit_wide = qrg_limit + wide_allow

        if self.anim_data:
            self.anim_ag_lines = []

        if self.worst_quality:
            log_y_s = self.log_sy
        else:
            log_y_s = self.log_y

        if self.basic_quality < QUALITY_BOUNDARY_LOW:
            log_y_e = self.log_sy
        else:
            log_y_e = self.log_y

        if self.too_long_peak_candidate is None:
            loop_stop = self.stop
        else:
            loop_stop = self.too_long_peak_candidate

        # print( 'make_cadidate_pairs: loop_stop=', loop_stop )

        pairs = []
        if self.fixed_guinier_start == 1:
            start_limit = self.peak + 1
        else:
            start_limit = loop_stop - MIN_GUINIER_SIZE
            if self.x2[start_limit] > MAX_START_Q2:
                # identify these cases
                max_start   = bisect_right( self.x2, MAX_START_Q2 )
                start_limit = max( self.peak + MIN_START_FROM_PEAK, max_start )

        # print( 'peak=', self.peak, 'start_limit=', start_limit )

        for start in range( self.peak, start_limit ):
            end     = start + MIN_GUINIER_SIZE
            e_slice = slice(end, self.stop)
            # print( 'start, end, stop=', start, end, self.stop )
            slope   = ( log_y_s[start] - log_y_e[e_slice] ) / ( self.x2[start] - self.x2[e_slice] )
            neg_i   = slope < 0
            neg_i_w = np.where( neg_i )[0]
            if len(neg_i_w) == 0:
                continue

            qrg     = self.x[e_slice][neg_i] * np.sqrt( -3*slope[neg_i] )
            try:
                j   = np.where( qrg < qrg_limit_wide )[0][-1]
            except:
                if False:
                    from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
                    etb = ExceptionTracebacker()
                    print( etb )
                continue

            stop = end + neg_i_w[j] + 1
            pairs.append( [ start, stop ] )

            if self.anim_data:
                to = stop - 1
                self.anim_ag_lines.append(  [ (self.x2[start], self.x2[to] ), (self.log_y[start], self.log_y[to]) ] )

            if self.qrg_stop is None or stop > self.qrg_stop:
                self.qrg_stop       = stop
                self.guinier_stop   = stop  # this usually will be improved later

        if self.anim_data:
            self.anim_guinier_stop  = self.qrg_stop

        # print( 'pairs=', pairs )
        return pairs

    def evaluate_interval( self, i, start, stop_a, qrg_limit, qrg_allow ):

        qrg_limit_ = qrg_limit + qrg_allow

        candidate_count = 0
        for stop in range( stop_a, start + MIN_GUINIER_SIZE, -1 ):
            size = stop - start
            if size < MIN_GUINIER_SIZE:
                break

            slice_ = slice( start, stop )

            x_ = self.x2[slice_]
            y_ = self.log_y[slice_]
            slope, intercept, r_value, p_value, stderr = reg_result = stats.linregress( x_, y_ )

            candidate_rec = None
            if slope <= 0:
                rg = compute_rg( slope )
                x = self.x[stop-1]
                qrg = x*rg
                # print( '(start, stop)=', (start, stop), 'qrg=', qrg, 'r_value=', r_value, 'end_consistency=', end_consistency )
                if qrg < qrg_limit_ or self.anim_data:
                    end_consistency = compute_end_consistency( start, stop, self.x2, self.log_y )
                    fwd_consistency = compute_fwd_consistency( start, stop, self.x2, self.log_y )
                    if self.worst_quality:
                        spline_slope = (self.log_sy[start] - self.log_sy[stop-1])/(x_[0] - x_[-1])
                        spline_rg = compute_rg( spline_slope )
                    else:
                        spline_rg = None
                    if USE_GUINIER_PLOT_LENGTH:
                        length = np.sqrt( (x_[0] - x_[-1])**2 + (y_[0] - y_[-1])**2 )
                    else:
                        length = size
                    interval_score, score_vector, score_array = evaluate_guinier_interval( self.basic_quality, self.px, self.dx, rg, length, r_value, end_consistency, fwd_consistency, spline_rg, return_vector=True )
                    candidate_rec = [ i, start, stop, reg_result, rg, end_consistency, interval_score, score_vector ]
                    if self.anim_data:
                        # anim_score = interval_score if qrg < qrg_limit_ else 0
                        anim_score = interval_score
                        anim_rec = [ start, stop, x_, y_, slope, intercept, anim_score, score_vector, rg ]
                        self.anim_cand_list.append( anim_rec )
                else:
                    interval_score = 0
            else:
                interval_score = 0
                qrg = 999

            if qrg < qrg_limit_ and candidate_rec is not None:
                if self.score is None or interval_score > self.score:
                    self.score     = interval_score
                    self.candidate = candidate_rec
                    if self.anim_data:
                        # evaluate_guinier_interval( self.basic_quality, self.px, self.dx, rg, length, r_value, end_consistency, fwd_consistency, spline_rg, debug=True )
                        self.best_history_dict[( start, stop )] = [ score_vector, rg ]
                candidate_count += 1

                if x_[0] < SMALL_Q2_BOUNDARY:
                    if self.smallq_score is None or interval_score > self.smallq_score:
                        self.smallq_score       = interval_score
                        self.smallq_candidate   = candidate_rec

                if candidate_count > 3:
                    break

    def guinier_interval( self, qrg_limit=1.3 ):
        """
        q*Rg < 1.3
        log_y = log(G) - x**2 * Rg**2/3
        slope = -Rg**2/3
        Rg = sqrt(-3*slope)
        """
        if self.anim_data:
            self.anim_cand_list = []

        # TODO: determine qrg_allow thoeoretically
        if self.worst_quality:
            qrg_allow = 1.0
        elif self.basic_quality > QUALITY_BOUNDARY_QRG:
            qrg_allow = 0
        elif self.basic_quality < QUALITY_BOUNDARY_LOW:
            qrg_allow = 0.5
        else:
            qrg_allow = 0.3     # required for 20160227-4

        self.guinier_stop   = self.stop  # this usually will be improved later
        self.candidate = None
        self.score     = None
        self.smallq_candidate   = None
        self.smallq_score       = None

        lq_candidate_count = 0
        wst_candidate_count = 0
        pairs = self.make_cadidate_pairs( qrg_limit )
        for i, pair in enumerate(pairs):
            self.evaluate_interval( i, pair[0], pair[1], qrg_limit, qrg_allow )
            if self.worst_quality:
                if self.candidate is not None:
                    wst_candidate_count += 1
                    if wst_candidate_count >= WST_AT_LEAST_TRIALS:
                        break
            elif self.basic_quality < QUALITY_BOUNDARY_LOW:
                if self.candidate is not None:
                    lq_candidate_count += 1
                    if lq_candidate_count >= LQ_AT_LEAST_TRIALS:
                        break

        if self.candidate is not None:
            # print( 'self.candidate=', self.candidate[0:3], self.candidate[4] )
            pass

        if self.candidate is None:
            self.set_guinier_result( self.make_low_quality_rec() )
        else:
            candidate = self.candidate
            start = candidate[1]
            if self.x2[start] >= SMALL_Q2_BOUNDARY:
                if self.smallq_score is not None:
                    score_ratio = self.smallq_score / self.score
                    if score_ratio > SCORE_RATIO_BOUNDARY:
                        candidate   = self.smallq_candidate

            self.set_guinier_result( candidate )

    def make_low_quality_rec(self, debug=False):
        if debug:
            print( 'making low_quality_rec' )
        start = 0
        stop = self.guinier_stop

        if self.qrg_stop is None:
            self.qrg_stop = stop
        if self.anim_guinier_stop is None:
            self.anim_guinier_stop = self.qrg_stop

        slice_ = slice( start, stop )
        x_ = self.x2[slice_]
        y_ = self.log_y[slice_]
        reg_result = stats.linregress( x_, y_ )
        return [ None, start, stop, reg_result, 0, 0, 0, ZERO_SCORE_VECTOR ]

    def set_guinier_result( self, candidate_rec ):
        self.guinier_start  = candidate_rec[1]
        self.guinier_stop   = candidate_rec[2]
        self.guinier_length = self.guinier_stop - self.guinier_start
        qunier_index = [ self.guinier_start, self.guinier_stop-1 ]
        self.guinier_x = self.x2[qunier_index]
        slope, intercept = candidate_rec[3][0:2]
        self.guinier_y = slope * self.guinier_x + intercept
        self.Rg = Rg = candidate_rec[4]
        self.Iz = np.exp( intercept )
        fr_ = self.guinier_start
        to_ = self.guinier_stop - 1
        self.min_q = self.x[fr_]
        self.min_qRg = self.min_q*Rg
        self.max_q = self.x[to_]
        self.max_qRg = self.max_q*Rg
        end_consistency_ = candidate_rec[5]

        e_size = END_GUINIER_SIZE*2
        if self.basic_quality < QUALITY_BOUNDARY_WORST or self.guinier_length < e_size:
            # as in 20160227-66
            # print( 'guinier_length=',  self.guinier_length )
            size_eval_limit         = ( min( e_size, self.guinier_length ) / e_size )**2
            quality_eval_limit      = min( self.basic_quality, QUALITY_BOUNDARY_WORST) / QUALITY_BOUNDARY_WORST
            eval_limit  = size_eval_limit * quality_eval_limit
            # print( 'eval_limit=', eval_limit )
            self.end_consistency    = min( eval_limit, end_consistency_ )
        else:
            self.end_consistency = end_consistency_
        self.r_value    = candidate_rec[3][2]
        self.score      = candidate_rec[6]
        self.score_vector = candidate_rec[7]

        if False:
            ret = compute_end_consistency( self.guinier_start, self.guinier_stop, self.x2, self.log_y, debug=True )
            print( 'ret=', ret )

        # print( 'set_guinier_result: r_value=', self.r_value, 'score=', self.score, 'max_qRg=', self.max_qRg, 'end_consistency=', self.end_consistency  )

    def set_guinier_null_result( self ):
        if False:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            raise RuntimeError( etb )
        self.guinier_x  = None
        self.guinier_y  = None
        self.Rg         = None
        self.Iz         = None
        self.min_q      = None
        self.min_qRg    = None
        self.max_q      = None
        self.max_qRg    = None
        self.end_consistency = 0
        self.basic_quality = 0
        self.score = 0
        self.score_vector = ZERO_SCORE_VECTOR
