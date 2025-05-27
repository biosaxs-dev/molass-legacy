"""
    Absorbance.py

    Copyright (c) 2017-2023, SAXS Tam, KEK-PF
"""
import numpy                as np
import copy
from scipy                  import stats
from bisect                 import bisect_right
import logging
from molass_legacy.KekLib.ChangeableLogger import arg_join
# from lmfit                  import minimize, Parameters
from molass_legacy.KekLib.LmfitThreadSafe import minimize, Parameters
from molass_legacy._MOLASS.SerialSettings import get_setting, INTEGRAL_BASELINE
from molass_legacy.SerialAnalyzer.ElutionCurve           import ElutionCurve
from molass_legacy.Baseline.ScatteringBaseline import ScatteringBaseline
from molass_legacy.SerialAnalyzer.ScatteringBasecurve import ScatteringBasecurve
from molass_legacy.SerialAnalyzer.ScatteringBasespline import ScatteringBasespline
from molass_legacy.SerialAnalyzer.BasePlane import LowPercentilePlane, LambertBeerPlane
from molass_legacy.SerialAnalyzer.UvBaseSurface import UvBaseSurface
from molass_legacy.Test.TesterLogger import write_to_tester_log
from molass_legacy.KekLib.ThreeDimUtils import compute_plane
import molass_legacy.KekLib.DebugPlot as plt

POSITVE_SMALL_VALUE     = 1e-5
MIN_ASC_WIDTH           = 10
USABLE_JUMP_POINT_RATIO = 0.1
LOW_PERCENTILE_OK_RATIO = 0.8       # not ok for HIF
LOW_PERCENTILE_BASE_1ST = 30
LOW_PERCENTILE_BASE_FIN = 10
USE_SUB_PICKING_IN_LPM  = True
END_ABNORMALITY_RATIO   = 5
ALLOWED_SHIFTED_RATIO   = 0.1
TAIL_BAND_WIDTH = 11

def compute_scatter( S, x, z ):
    denom = np.max( np.vstack( [ np.ones( len(x) )*1e-10, 1 - S * np.power( x, -4 ) ] ), axis=0 )
    return np.dot( np.log( 1 / denom ).reshape( ( len(x), 1 ) ), z.reshape( ( 1, len(z) ) ) )

def non_zero_value( v ):
    if v >= 0:
        v_ = max( POSITVE_SMALL_VALUE, v )
    else:
        v_ = min( -POSITVE_SMALL_VALUE, v )
    return v_

class FixedBaseResult:
    def __init__( self, fidex_base, fixed_slope, size, j ):

        if fixed_slope is None:
            A   = 0
            B   = 0
            C   = fidex_base
        else:
            A   = 0
            B   = fixed_slope
            j_  = 0  if j is None else j
            C   = fidex_base - fixed_slope * j_

        self.params  = params = Parameters()
        params.add( 'A', value=A, vary=False )
        params.add( 'B', value=B, vary=False )
        params.add( 'C', value=C, vary=False )
        params.add( 'S', value=0, vary=False )

class Absorbance:
    def __init__( self, wl_vector, data, x_curve,
            col_header=None,
            std_wvlen=None, end_wvlen=None,
            scattering_base=None, independent=False, use_lpm=False,
            orig_top_x=None,
            debug=False ):
        self.debug      = debug
        self.logger     = logging.getLogger( __name__ )
        wl_slice_ = self.check_abnormality_in_wvlen_ends(data, wl_vector)
        self.wl_vector  = wl_vector[wl_slice_]
        self.data       = data[wl_slice_, :]
        """
        Due to thie slicing,
        be aware that self.data is not always equals to the input data,
        i.e., serial_data.conc_array.
        """
        self.x_curve    = x_curve
        self.orig_top_x = orig_top_x
        self.col_header = col_header
        self.better_mapping_info = None
        if scattering_base is None:
            scattering_base = get_setting( 'scattering_base' )
        # self.independent = ( independent or scattering_base == 0 )
        self.independent = independent
        self.rail_points    = None

        self.jvector    = np.arange( self.data.shape[1] )
        self.update_optimizer_range( std_wvlen, end_wvlen )
        self.mapped_intensity = None
        # self.compute_base_curve( use_lpm )
        # write_to_tester_log( 'real_flow_changes=' + str( self.get_real_flow_changes() ) + '\n' )

        # print( 'Absorbance: jump_j=', self.jump_j )

    def update_optimizer_range( self, std_wvlen, end_wvlen ):
        from molass_legacy.UV.PlainCurveUtils import get_flat_wavelength

        self.std_wvlen  = get_setting( 'absorbance_picking' ) if std_wvlen is None else std_wvlen
        self.end_wvlen  = get_flat_wavelength(self.wl_vector) if end_wvlen is None else end_wvlen
        self.reg_wvlens = np.linspace( self.std_wvlen, self.end_wvlen, 4 )
        # print( 'reg_wvlens=', self.reg_wvlens )

        a_vector, i     = self.get_vector_at( self.std_wvlen )
        self.index      = i
        self.a_vector   = a_vector
        self.istd       = bisect_right( self.wl_vector, self.std_wvlen )
        if USE_SUB_PICKING_IN_LPM:
            self.istd_sub   = bisect_right( self.wl_vector, get_setting('absorbance_picking_sub') )
        self.iend       = bisect_right( self.wl_vector, self.end_wvlen )

        # take min to copw with cases where wl_vector has been trimmed
        self.i400       = min(len(self.wl_vector) - TAIL_BAND_WIDTH, bisect_right(self.wl_vector, 400))

        self.a_curve    =  ElutionCurve( self.a_vector, orig_top_x=self.orig_top_x )

        wvlen_sub = get_setting( 'absorbance_picking_sub' )
        a_vector_sub, i_sub = self.get_vector_at( wvlen_sub )
        self.index_sub = i_sub
        self.a_vector_sub = a_vector_sub
        self.a_curve_sub =  ElutionCurve( self.a_vector_sub, possiblly_peakless=True )

    def check_abnormality_in_wvlen_ends( self, data, wl_vector):
        num_points = 5
        start_sumv_array = np.array([ np.sum(np.abs(data[i,:])) for i in range(num_points) ])
        average_len = 3

        self.safe_start = None
        last_average = None
        for k in range(num_points - average_len + 1):
            average = np.average(start_sumv_array[k:k+average_len])
            if last_average is not None:
                ratio = last_average / average
                # print([k], ratio)
                if ratio > END_ABNORMALITY_RATIO:
                    self.safe_start = k

            last_average = average
        if self.safe_start is not None:
            i = self.safe_start - 1
            self.logger.warning("abnormal values detected at wave length[%d]=%g" % (i, wl_vector[i]))

        size = data.shape[0]
        final_sumv_array = np.array([ np.sum(np.abs(data[i,:])) for i in range(size-num_points, size) ])
        self.safe_end = None
        last_average = None
        for k in reversed(range(num_points - average_len + 1)):
            average = np.average(final_sumv_array[k:k+average_len])
            if last_average is not None:
                ratio = last_average / average
                # print([k], ratio)
                if ratio > END_ABNORMALITY_RATIO:
                    self.safe_end = size - num_points + k
            last_average = average

        if self.safe_end is not None:
            i = self.safe_end + 1
            self.logger.warning("abnormal values detected at wave length[%d]=%g" % (i, wl_vector[i]))

        return slice(self.safe_start, self.safe_end)

    def get_wave_len_ends( self ):
        return self.safe_start, self.safe_end

    def get_shifted_elution_base( self ):
        # compute_base_curve must have been called for self.rail_points
        fc = self.rail_points
        m_base  = np.zeros( fc[1]-fc[0]+1 ) if self.base_curve is None else self.base_curve
        l_base  = np.ones( fc[0] ) * m_base[0]
        r_base  = np.ones( len(self.a_vector) - fc[1] -1 ) * m_base[-1]
        base    = np.hstack( [ l_base, m_base, r_base ] )

        if False:
            print("len(base), len(self.a_vector)", len(base), len(self.a_vector))
            plt.push()
            fig, ax = plt.subplots()
            ax.plot(self.a_vector)
            ax.plot(base)
            plt.show()
            plt.pop()

        assert len(base) == len(self.a_vector)
        return base

    def get_standard_elution_base( self ):
        return self.get_bottomline_vector()

    def compute_base_curve( self, pre_recog, baseline_type, full_width=True, debug=False ):
        """
        full_width=True because this baseline_type should be called after trimming
        """

        if full_width:
            # temp fix to avoid getting false j_min, j_max
            j_min, j_max = 0, len(self.a_vector)-1
        else:
            fc = pre_recog.flowchange
            j_min, j_max1 = fc.get_flow_changes()
            # j_max1 can be the length
            j_max = j_max1 - 1

        self.logger.info('computing base curve (type %d)  for UV data with j_min=%d, j_max=%d in data.shape=%s' % (baseline_type, j_min, j_max, str(self.data.shape)))

        base_curve_done = False

        if baseline_type == 0:
            self.base_curve = np.zeros(j_max + 1 - j_min)
            self.baseplane_params = None
            base_curve_done = True
        elif baseline_type == 1:
            self.solve_bottomplane_LPM(j_min, j_max, debug=debug)
        elif baseline_type == 4:
            self.solve_bottomplane_LB(j_min, j_max, debug=debug)
        elif baseline_type == 5:
            self.compute_integral_basecurve()
            base_curve_done = True
        else:
            assert False

        if base_curve_done:
            return

        self.rail_points    = [ j_min, j_max ]

        # print( 'compute_base_curve: len(self.a_curve.peak_info)=', len(self.a_curve.peak_info) )

        if False:
            self.rail_points_sigma  = [ self.a_curve.compute_end_coordinate_in_sigma(i) for i in self.rail_points ]
            print( 'rail_points_sigma=', self.rail_points_sigma )
        else:
            self.rail_points_sigma  = None

        A, B, C = self.get_baseplane_params()
        estimeted_rails_base    = [ A*self.std_wvlen + B*i + C  for i in self.rail_points ]
        """
        self.rail_points_offset = [ ( self.a_curve.y[i] - ( A*self.std_wvlen + B*i + C ) ) / self.a_curve.max_y for i in self.rail_points ]
        print( 'rail_points_offset=', self.rail_points_offset )
        """

        self.tail_band      = slice( self.i400, self.i400+TAIL_BAND_WIDTH )
        self.tail_index     = self.i400 + 5
        self.tail_curve     = np.average( self.data[ self.tail_band, j_min:j_max+1 ], axis=0 )
        rails = [
                self.data[ [ self.index, self.tail_index ], j_min ],
                self.data[ [ self.index, self.tail_index ], j_max ],
                ]
        try:
            base_surface  = UvBaseSurface( self.rail_points, self.tail_curve, rails,
                                        self.rail_points_sigma,
                                        self.a_curve.max_y,
                                        estimeted_rails_base,
                                        )
            self.base_curve = base_surface.get_basecurve()
        except RuntimeError:
            self.base_curve = None

        return self.base_curve

    def get_integral_basecurve(self):
        return self.base_curve

    def get_vector_at( self, wvlen ):
        i = min(len(self.wl_vector) - 1, bisect_right( self.wl_vector, wvlen ))
        return np.array( self.data[i, :] ), i

    def get_standard_vector( self ):
        return self.a_vector

    def get_corrected_vector( self, absorbance_baseline_type, scattering_base, baseline_degree, params=None ):
        slice_size = self.right_jump_j_safe - self.jump_j_safe + 1

        # scattering_base = get_setting( 'scattering_base' )
        # baseline_degree = get_setting( 'baseline_degree' )
        # absorbance_baseline_type = get_setting( 'absorbance_baseline_type' )

        if absorbance_baseline_type == 0 or self.base_curve is None:
            if scattering_base == 2 and slice_size / len(self.a_vector) > LOW_PERCENTILE_OK_RATIO:
                a_  = self.a_vector[self.jump_j_safe:self.right_jump_j_safe+1]
                j   = self.jvector - self.jump_j_safe
                if baseline_degree == 1:
                    abl = ScatteringBaseline( a_ )
                    A, B = abl.solve()
                    self.bottomline = A*j + B
                elif baseline_degree == 2:
                    abl = ScatteringBasecurve( a_ )
                    A, B, C = abl.solve()
                    self.bottomline = A*j**2 + B*j + C
                else:
                    abl = ScatteringBasespline( a_ )
                    abl.solve()
                    self.bottomline = abl.get_baseline( j )
            else:
                self.bottomline = self.get_bottomline_vector( params=params )
        else:
            self.bottomline = np.hstack( [
                            np.ones( self.jump_j_safe ) * self.base_curve[0],
                            self.base_curve,
                            np.ones( len(self.a_curve.y) - 1 - self.right_jump_j_safe ) * self.base_curve[-1],
                            ] )

        y = self.a_vector - self.bottomline
        y[ y < 0 ] = 0
        y[ 0:self.jump_j_safe ] = 0
        return y

    def compute_scatter_factor_at( self, A, B, C, i ):
        peak_j  = self.a_curve.middle_abs
        end_wvlen_ = self.wl_vector[i]
        std_base = A * self.std_wvlen + B * peak_j + C
        end_base = A * end_wvlen_ + B * peak_j + C

        """
            Abs_ = log( 1 / ( 1 - S*λ⁻⁴ ) ) * Abs0
            exp( Abs_ / Abs0 ) = 1 / ( 1 - S*λ⁻⁴ )
            1 - S*λ⁻⁴ = 1 / exp( Abs_ / Abs0 ) = exp( - Abs_ / Abs0 )
            1 - exp( - Abs_ / Abs0 ) = S*λ⁻⁴
            S = (  1 - exp( - Abs_ / Abs0 ) ) * λ⁴
        """
        S = ( 1 - np.exp( - (self.data[ i, peak_j ] - end_base )
                                    / ( self.a_vector[peak_j] - std_base )
                              )
            ) * np.power( end_wvlen_, 4 )
        return S

    def get_jump_point_base( self ):
        A, B, C = self.get_baseplane_params()
        z = A*self.std_wvlen + B*self.jump_j_safe + C
        return z

    def get_mapped_slope_and_intercept( self ):
        A, B, C = self.get_baseplane_params()
        slope     = B
        intercept = A * self.std_wvlen + C
        return slope, intercept

    def get_conc_params( self, conc_factor ):
        A, B, C = self.get_baseplane_params()
        conc_slope     = conc_factor * B
        conc_intercept = conc_factor * ( A * self.std_wvlen + C )
        print( __name__ + '.get_conc_params: conc_slope=', conc_slope, 'conc_intercept=', conc_intercept )
        return conc_slope, conc_intercept

    def get_bottomline_vector( self, params=None ):
        if params is None:
            A, B, C = self.get_baseplane_params()
        else:
            A, B, C = params['A'], params['B'], params['C']
        # print( 'get_bottomline_vector: S=', S.value )
        jv  = self.jvector
        base =  A * self.std_wvlen + B * jv + C
        if False:
            print( 'get_bottomline_vector(%d): A=%.4g, B=%.4g, C=%.4g' % (id(self), A, B, C ), params)
            fig = plt.figure()
            ax = fig.gca()
            ax.set_title("get_bottomline_vector: debug")
            ax.plot(jv, self.a_vector, color='blue')
            ax.plot(jv, base, color='red')
            plt.show()
        return base

    def get_bottomline_vector_near_peak( self ):
        A, B, C = self.get_baseplane_params()
        lower   = self.a_curve.lower_abs
        middle  = self.a_curve.middle_abs
        upper   = self.a_curve.upper_abs
        jv  = np.array( [ lower, middle, upper ] )
        ret_vectors = compute_plane( A, B, C, self.reg_wvlens, jv )
        z0 = self.a_vector[jv] - ret_vectors[0,:]
        ret_scatter = compute_scatter( S, self.reg_wvlens, z0 )

        return ( ret_vectors + ret_scatter ).T

    def get_wv_index_vector( self ):
        iv = []
        for wvlen in self.reg_wvlens:
            iv.append( bisect_right( self.wl_vector, wvlen ) )
        self.index_vector = np.array( iv )
        return self.index_vector

    def get_bottomline_matrix( self ):
        A, B, C = self.get_baseplane_params()
        jv  = self.jvector
        ret_matrix  = compute_plane( A, B, C, self.reg_wvlens, jv )
        z0 = self.a_vector - ret_matrix[0,:]
        ret_scatter = compute_scatter( 0, self.reg_wvlens, z0 )
        return ( ret_matrix + ret_scatter).T

    def get_normal_region( self ):
        return self.jump_j_safe, self.right_jump_j_safe

    def solve_bottomplane_LPM( self, j_min, j_max, debug=False ):
        iy  = slice( j_min, j_max+1 )
        uv_lpm_option = get_setting('uv_lpm_option')
        # suppress 3D-LPM temporarily
        if uv_lpm_option == 1 and False:
            if USE_SUB_PICKING_IN_LPM:
                start = self.istd_sub
            else:
                start = self.istd
            ix  = slice( start, self.iend )
            lpp = LowPercentilePlane(self.data, self.wl_vector, self.a_vector, ix, iy, self.index, debug)
            params = lpp.get_params()
        else:
            from molass_legacy.SerialAnalyzer.Uv2dLpm import Uv2dLpm
            lpm = Uv2dLpm(self.a_vector, iy)
            params = lpm.get_params()

        option_text = '3D' if uv_lpm_option == 1 else '2D'
        self.logger.info('UV baseplane_params are determined to %s with %s option.', str(params), option_text)
        self.baseplane_params = params

    def get_baseplane_params( self ):
        return self.baseplane_params

    def solve_bottomplane_LB( self, j_min, j_max,
            init_values=None, fixed_base=None, fixed_slope=None, left_end_j=None,
            debug=False ):

        if USE_SUB_PICKING_IN_LPM:
            start = self.istd_sub
        else:
            start = self.istd
        ix  = slice( start, self.iend )
        iy  = slice( j_min, j_max+1 )
        istd_ = self.istd - ix.start

        lbp = LambertBeerPlane(self.data, self.wl_vector, ix, iy, self.a_vector, self.index, istd_, debug)
        self.baseplane_params   = lbp.get_params()
        print('solve_bottomplane_LB(%d)' % id(self), self.baseplane_params)

    def get_corrected_data( self ):
        A, B, C = self.baseplane_params
        x   = self.wl_vector 
        y   = self.jvector
        base_plane = compute_plane( A, B, C, x, y )
        data_ = self.data - base_plane
        return data_

    def get_zlim_for_3d_plot( self ):
        start = bisect_right( self.wl_vector, 210 )
        ix  = slice( start, self.iend+1 )
        iy  = slice( self.jump_j_safe, self.right_jump_j_safe+1 )
        data_ = self.data[ix, iy]
        zmin = np.min(data_)
        zmax = np.max(data_)
        return zmin, zmax

    def compute_integral_basecurve(self):
        from molass_legacy.Baseline.Baseline import better_integrative_curve
        end_slices = self.a_curve.get_end_slices()
        self.base_curve, convex = better_integrative_curve(self.a_vector, end_slices=end_slices)
        self.logger.info("computed UV basecurve with better_integrative_curve%s.", " using convex method" if convex else "")
        return self.base_curve

    def shifted_baseline_ok(self):
        lower, center, upper = self.a_curve.get_primary_peak_info()
        peak_top_ratio = self.a_curve_sub.y[center]/self.a_curve.y[center]
        ret = peak_top_ratio < ALLOWED_SHIFTED_RATIO
        if not ret:
            self.logger.info("peak_top_ratio=%.3g at λ=%g over λ=%g.", peak_top_ratio, self.end_wvlen, self.std_wvlen)
        return ret
