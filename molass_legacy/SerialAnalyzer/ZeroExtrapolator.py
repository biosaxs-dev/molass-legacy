"""

    ZeroExtrapolator.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF
"""
import time
import numpy                as np
import logging
from scipy                  import stats
# from lmfit                  import minimize, Parameters
from molass_legacy.KekLib.LmfitThreadSafe import minimize, Parameters
from molass_legacy.KekLib.BasicUtils import ordinal_str
import molass_legacy.KekLib.OurStatsModels as sm
from molass_legacy.AutorgKek.DataModels import GuinierPorodLmfit
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting
from molass_legacy.KekLib.ProgressInfo import on_stop_raise
# from molass_legacy.LRF.LrfInfo import LrfInfo

NUM_EXTRAPOLATION_POINTS    = 5
USE_ERROR_PROPAGATION   = True
USE_MOORE_PENROSE_ONLY  = False
USE_LRF_RESULT_POOL = True
MIN_G_LN    = -100
MIN_G       = np.exp( MIN_G_LN )
IE_MIN_WIDTH        = 30
IE_B_MAX_MAX        = 0.1
IE_B_MIN_MAX        = 0.01
IE_B_ROUGH_BOUNDARY = 0.1
IE_P_VALUE_BOUNDARY = 0.5
NEXT_TEST_SPAN      = 5
RESET_SPAN          = 50
DYNAMIC_TEST_OK_COUNT   = 3
SEE_AQ_DEVIATION    = False
ACCEPTABLE_AQ_DEVIATION = 0.01
TO_RECOMPUTE_BQ_DEVIATION = 0.0001

def is_boundary_reg_params( reg_params ):
    # return False
    if reg_params is None:
        return False    # meaning not a cadidate

    r_value, p_value, std_err = reg_params
    return r_value > -0.95 or p_value > IE_P_VALUE_BOUNDARY or std_err > 5

def is_boundary_candidate( max_y, y, slope, p_value, j=None, log_aq_dev=None ):
    max_y_ = max( IE_B_MIN_MAX, min( IE_B_MAX_MAX, max_y ) )
    y_average = abs( np.average( y[-IE_MIN_WIDTH:] ) ) / max_y_

    if False and ( j is None or j > 100 and j < 300 ):
         print( [j], 'y_average=', y_average , 'slope=', slope, 'p_value=', p_value )

    if log_aq_dev is not None and log_aq_dev > ACCEPTABLE_AQ_DEVIATION:
        return True

    if y_average < IE_B_ROUGH_BOUNDARY:
        return p_value > IE_P_VALUE_BOUNDARY
    else:
        return False

def is_boundary_candidate_simple( y, slope, p_value ):
    return (    abs( np.average( y[-20:] ) ) < IE_B_ROUGH_BOUNDARY
            and abs( slope ) < IE_SLOPE_BOUNDARY
            and p_value > IE_P_VALUE_BOUNDARY )

def make_cx_vector_impl( use_elution_models, conc_factor, ad, paired_range, indeces, c_vector, jvector ):
    if use_elution_models:
        """
        temporary fix using jvector to avoid emg rare case trouble
        """
        wc_vector = np.zeros(len(jvector))
        for elm_rec in paired_range.elm_recs:
            e   = elm_rec[0]
            fnc = elm_rec[1]
            wc_vector += fnc(jvector)
        wc_vector *= conc_factor
        c_vector = wc_vector[indeces]

    zero_c  = 0.0
    min_c   = np.min( c_vector )
    max_c   = np.max( c_vector )
    min_c_1 = min_c * 4/5

    if ad == 0:
        cx_vector = np.hstack( [ np.linspace( zero_c, min_c_1, NUM_EXTRAPOLATION_POINTS ), c_vector ] )
    else:
        cx_vector = np.hstack( [ c_vector, np.linspace( min_c_1, zero_c, NUM_EXTRAPOLATION_POINTS ) ] )

    return cx_vector, min_c, max_c

class ZeroExtrapolator:
    def __init__( self, qvector, preview_params, serial_data, mapped_info, applied_ranges, known_info_list=None ):
        self.logger     = logging.getLogger( __name__ )
        self.qvector    = qvector
        self.sd         = serial_data
        self.mapped_info    = mapped_info
        self.applied_ranges = applied_ranges
        self.known_info_list = known_info_list
        self.conc_factor    = serial_data.conc_factor
        self.logger.info("lrf executer has been constructed with conc_factor=%g.", self.conc_factor)
        self.zx_boundary        = get_setting( 'zx_boundary' )
        self.num_q_points       = get_setting( 'zx_num_q_points' )
        self.zx_build_method    = get_setting( 'zx_build_method' )
        self.zx_boundary_method = get_setting( 'zx_boundary_method' )
        self.ignore_all_bqs     = get_setting('ignore_all_bqs')
        self.ignore_bq_list     = get_setting('ignore_bq_list')
        self.recompute_regboundary  = get_dev_setting( 'recompute_regboundary' )

        pdata, popts = preview_params
        self.use_elution_models = popts.use_elution_models

        if USE_LRF_RESULT_POOL:
            from molass_legacy.LRF.LrfResultPool import LrfResultPool
            self.pool = LrfResultPool(pdata, popts)
            self.pool.run_solver()
            self.logger.info("ZeroExtrapolator: using LRF Ppol.")
        else:
            assert False

        self.suppress_defer_test    = get_dev_setting( 'suppress_defer_test' )
        self.zx_add_constant        = get_dev_setting( 'zx_add_constant' )
        if self.zx_add_constant == 1:
            self.logger.info( 'Constant term will be added to the regression formulation.' )

    def make_cx_vector( self, ad, paired_range, indeces, c_vector ):
        return make_cx_vector_impl(self.use_elution_models, self.conc_factor, ad, paired_range, indeces, c_vector, self.sd.jvector)

    def extrapolate( self, m, ad, range_no, intensities, c_vector, cx_vector, max_c, boundary_q=0.01, guinier_boundary=None, temp_folder=None, debug_plot=False, debug_save=False ):
        self.stop_check()

        ones = np.ones( len(cx_vector) )
        # I(s) = C * A(s) + C**2 * B(s)
        # I(s)/C = A(s) + C * B(s)
        cx_matrix = np.array( [ ones, cx_vector ] ).T

        # self.logger.info( 'debug: guinier_boundary=' + str(guinier_boundary) )
        self.guinier_boundary   = guinier_boundary
        self.add_reg_candidate  = True

        result_with_reg, lrf_info = self.extrapolate_impl_MP_inverse( m, ad, range_no, intensities, c_vector, cx_vector, max_c, cx_matrix, boundary_q )

        return result_with_reg[0:-1], lrf_info

    def do_debug_save( self, m, ad, result, temp_folder ):
        if self.zx_boundary_method != 'AUTO' or self.guinier_boundary is None or temp_folder is None:
            return

        from molass_legacy.KekLib.NumpyUtils import np_savetxt

        aq_latest_list  = result[-1]
        aq_latest_array = np.array( aq_latest_list )
        start = self.guinier_boundary+1
        slice_ = slice(start,start+len(aq_latest_array))
        x_ = self.qvector[slice_]
        aq_latest_file = temp_folder + '/aq_latest_array-%d-%d.csv' % (m, ad)
        print( x_.shape, aq_latest_array.shape )
        np_savetxt( aq_latest_file, np.vstack( [ x_, aq_latest_array.T ] ).T )

    def do_debug_plot( self, results ):
        if self.zx_boundary_method != 'AUTO' or self.guinier_boundary is None:
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,8))

        max_lim_list = []

        param_types = [ '(no reg)', '(with reg)' ]

        for i, result in enumerate( results ):
            aq_latest_list  = result[-1]
            boundary_j      = result[-2]

            aq_latest_array = np.array( aq_latest_list )
            start = self.guinier_boundary+1
            slice_ = slice(start,start+len(aq_latest_array))
            x_ = self.qvector[slice_]

            for k, item in enumerate(['r_value', 'p_value', 'stderr']):
                ax = axes[i, k]
                ax.set_title( param_types[i] + ' ln(A(q)) 30p-segment ' + item )
                y_ = aq_latest_array[:,k]
                ax.plot( x_, y_ )
                if boundary_j is not None:
                    j_ = boundary_j - start
                    if j_ >= 0 and j_ < len(x_):
                        ax.plot( x_[j_], y_[j_], 'or' )

                if i == 0:
                    max_lim_list.append( [ ax.get_xlim(), ax.get_ylim() ] )
                else:
                    ax.set_xlim( max_lim_list[k][0] )
                    ax.set_ylim( max_lim_list[k][1] )

        plt.tight_layout()
        plt.show()

    def extrapolate_impl_MP_inverse( self, m, ad, range_no, intensities, c_vector, cx_vector, max_c, cx_matrix, boundary_q  ):
        self.boudary_test_deferred  = False
        self.boundary_j = None      # to be removed
        self.bq_max = 0             # to be removed

        ones = np.ones( len(cx_vector) )
        # I(s) = C * A(s) + C**2 * B(s)
        # I(s)/C = A(s) + C * B(s)
        cx_matrix = np.array( [ ones, cx_vector ] ).T

        data_list = []
        error_list = []
        for intensity in intensities:
            data_list.append( intensity.orig_array[:,1] )
            error_list.append( intensity.orig_array[:,2] )

        self.aq_max_c = data_list[-1 if ad == 0 else 0] / max_c
        self.log_aq_span = abs( np.log( np.average( self.aq_max_c[10:20]/self.aq_max_c[-20:-10] ) ) )

        data_matrix = np.vstack( data_list )
        error_matrix = np.vstack( error_list )
        print( 'data_matrix.shape=',data_matrix.shape )

        self.stop_check()

        C   = c_vector
        C2  = C*C
        C_  = np.array( [C, C2] ).T
        Cpinv = np.linalg.pinv( C_ )        # Moore-Penrose inverse

        if USE_MOORE_PENROSE_ONLY:
            P   = np.dot( Cpinv, data_matrix )
        else:
            if USE_LRF_RESULT_POOL:
                A, B, Z, E, lrf_info, C = self.pool.solver_results[range_no]
                P = np.vstack( [A, B] )
            else:
                assert False

        """
        
        """
        ze_array = np.dot( cx_matrix, P ).T
        param_array = P.T
        # ab_error_array = np.sqrt( np.dot( Cpinv**2, error_matrix**2 ) ).T
        ab_error_array = np.array(E[0:2]).T

        # errors for I(s) = C * A(s) + C**2 * B(s)
        ze_error_array = np.sqrt( np.dot( ab_error_array**2, cx_matrix.T**2, ) )

        if False:
            self.logger.info("ze_array.shape=%s, param_array.shape=%s, ab_err___array.shape=%s, ze_err___array.shape=%s",
                            str(ze_array.shape), str(param_array.shape), str(ab_error_array.shape), str(ze_error_array.shape))

        aq_latest_list = []
        return [data_matrix.T/c_vector, ze_array, param_array, ab_error_array, ze_error_array, self.boundary_j, aq_latest_list], lrf_info

    def dump_data(self, m, ad, C):
        import os
        from molass_legacy.KekLib.NumpyUtils import np_savetxt
        analysis_folder = get_setting('analysis_folder')
        file = os.path.join(analysis_folder, "C-%d-%d.dat" % (m, ad))
        np_savetxt(file, C.T)

    def stop_check( self ):
        on_stop_raise()

class GuinierPorodAnalyzer:
    def __init__( self, Q, I ):
        self.x  = Q**2
        self.y  = np.log(I)
        self.y[ I <= 0 ] = MIN_G_LN
        self.Q  = Q
        self.I  = I
        """
        I_positive = I > 0
        Q_ = Q[I_positive]
        I_ = I[I_positive]
        self.x  = Q_**2
        self.y  = np.log(I_)
        self.Q  = Q_
        self.I  = I_
        """

    def fit( self ):
        slope, intercept, r_value, p_value, std_err = stats.linregress( self.x, self.y )
        Rg  = np.sqrt( 3*(-slope) ) if slope < 0 else None
        G   = max( MIN_G, np.exp( intercept ) )
        d   = 2.5

        guinier_porod = GuinierPorodLmfit()

        params = Parameters()
        params.add('G',     value= G,  min=0 )
        params.add('Rg',    value= Rg, min=1.0, max=500.0 )
        params.add('d',     value= d,  min=1,   max=4 )

        result = minimize( guinier_porod, params, args=( self.Q, self.I, 1 ) )
        G_, Rg_, d_ = result.params['G'].value, result.params['Rg'].value, result.params['d'].value
        G_stdev = 0
        Rg_stdev = 0

        return G_, G_stdev, Rg_, Rg_stdev
