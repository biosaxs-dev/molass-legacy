"""

    ScatteringBaseCorrector.py

    Copyright (c) 2017-2024, SAXS Team, KEK-PF

"""
import numpy                as np
import copy
import logging
from matplotlib.patches     import Polygon
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from molass_legacy._MOLASS.SerialSettings         import get_setting, get_xray_picking, XARY_BASE_LINEAR, XARY_BASE_QUADRATIC, XARY_BASE_SPLINED
from molass_legacy.SerialAnalyzer.ElutionBaseCurve       import ElutionBaseCurve
from molass_legacy.Baseline.ScatteringBaseline     import ScatteringBaseline
from ScatteringBasecurve    import ScatteringBasecurve
from ScatteringBasespline   import ScatteringBasespline
from molass_legacy.KekLib.Affine                 import Affine
from DebugCanvas            import DebugCanvas
from DevSettings            import get_dev_setting

DEBUG_PLOT              = False
VERIY_SMALL_ANGLE_LIMIT = 0.02
SMALL_ANGLE_LIMIT       = 0.1
USE_CONSTANT_PERCENTILE = True
ENABLE_ZERO_BASE        = True
HPM_RATIO               = 0.8
BETTER_INTEGRATIVE_CURVE = True

if USE_CONSTANT_PERCENTILE:
    from scipy.interpolate  import LSQUnivariateSpline

from BasePercentileOffset   import base_percentile_offset

def compute_baseline_using_LPM_impl(baseline_type, num_iterations, i, y, p_final=None, k=None, curve=None, logger=None, suppress_log=True, debug=False):
    """
    moved due to:
        ImportError: cannot import name 'LPM_PERCENTILE' from partially initialized module 'Baseline.Baseline' (most likely due to a circular import)
    """
    from molass_legacy.Baseline.Baseline import LPM_PERCENTILE, integrative_curve, better_integrative_curve

    baseline_total  = np.zeros(len(y))

    if baseline_type == 5:
        integral = True
        orig_y = y.copy()
    else:
        integral = False

    if not suppress_log:
        logger.info("computing the baseline with baseline_type=%d", baseline_type)

    if baseline_type > XARY_BASE_SPLINED:
        # temporary fix when called in LPM+MF
        baseline_type = XARY_BASE_LINEAR

    if debug:
        if k is not None and k < 10:
            print( 'LPM_impl', [k] )

    if p_final is None:
        p_final = LPM_PERCENTILE

    if baseline_type < 4:
        baseline_type_exec = baseline_type
    else:
        baseline_type_exec = XARY_BASE_LINEAR

    for k in range( num_iterations ):
        if baseline_type_exec == XARY_BASE_LINEAR:
            # x requiring warning is intentionally suppressed here fot the time being
            sbl = ScatteringBaseline( y, curve=curve, logger=logger, suppress_warning=True )
            D, E  = sbl.solve( p_final=p_final )
            xray_baseline   = D*i + E
        elif baseline_type_exec == XARY_BASE_QUADRATIC:
            sbl = ScatteringBasecurve( y )
            F, D, E = sbl.solve( p_final=p_final )
            xray_baseline   = F*i**2 + D*i + E
        else:
            assert( baseline_type_exec == XARY_BASE_SPLINED )
            sbl = ScatteringBasespline( y )
            F, D, E = sbl.solve( p_final=p_final )
            xray_baseline   = sbl.get_baseline( i )

        y -= xray_baseline
        baseline_total += xray_baseline

    if integral:
        if BETTER_INTEGRATIVE_CURVE:
            end_slices = curve.get_end_slices()
            temp_baseline, convex = better_integrative_curve(orig_y, baseline_total, p_final, end_slices=end_slices)
            if convex and logger is not None:
                logger.info("integrative curve has been made using convex method.")
        else:
            temp_baseline = integrative_curve(orig_y, baseline_total)
        diff_baseline = temp_baseline - baseline_total
        y -= diff_baseline
        baseline_total = temp_baseline

    if False:
        import molass_legacy.KekLib.DebugPlot as plt
        plt.plot( y )
        plt.plot( xray_baseline )
        plt.show()

    return baseline_total

class ScatteringBaseCorrector:
    def __init__( self, jvector, qvector, intensity_array,
                    curve=None,
                    index=None,
                    affine_info=None,
                    inty_curve_y=None,
                    baseline_opt=None,
                    baseline_type=None,
                    need_adjustment=None,
                    parent=None, with_demo=False,):

        self.logger     = logging.getLogger( __name__ )
        self.parent     = parent
        self.show_plot  = DEBUG_PLOT or with_demo
        self.qvector    = qvector
        self.jvector    = jvector
        self.intensity_array    = intensity_array
        self.curve      = curve
        self.x_vector   = affine_info[0]
        self.x_base     = affine_info[1]
        self.x_adjust   = affine_info[2]
        npp = int( len(self.x_vector)*HPM_RATIO )
        self.hpm_i      = sorted( np.argpartition( self.x_vector, npp )[npp:] )
        self.top_x      = np.average( self.hpm_i )
        self.hpm_y      = self.x_vector[self.hpm_i]
        self.top_y      = np.average( self.hpm_y )
        self.inty_curve_y   = inty_curve_y
        self.lpm_params = None

        if baseline_opt is None:
            baseline_opt    = get_setting( 'xray_baseline_opt' )

        if baseline_type is None:
            baseline_type = get_setting( 'xray_baseline_type' )

        self.logger.info("ScatteringBaseCorrector starts with baseline_type=%d", baseline_type)

        if need_adjustment is None:
            need_adjustment = get_setting( 'xray_baseline_adjust' ) == 1

        knots   = np.linspace( 0, len(jvector), len(jvector)//10 )
        self.iknots  = knots[1:-1]

        ecurve = ElutionBaseCurve( inty_curve_y )
        self.size_sigma = ecurve.compute_size_sigma()
        self.logger.info( 'size_sigma=%g' % ( self.size_sigma ) )

        self.baseline_opt       = baseline_opt
        self.baseline_type = baseline_type
        if baseline_type == 5:
            end_slices = get_setting('manual_end_slices')
        else:
            end_slices = None
        if end_slices is None:
            end_slices = curve.get_end_slices()
        self.end_slices = end_slices
        self.need_adjustment    = need_adjustment
        self.num_iterations     = get_setting( 'correction_iteration' )
        self.log_xray_lpm_params = get_dev_setting('log_xray_lpm_params')

    def correct_with_matrix_base( self, mapped_info, progress_cb=None, return_base=False ):
        from molass_legacy.Baseline.BaseScattering import basescattering_correct

        self.logger.info("correct_with_matrix_base")

        # use_mpi=True is not yet implemented properly
        base = basescattering_correct(mapped_info, self.intensity_array, use_mpi=False, end_slices=self.end_slices)
        if return_base:
            return base

    def correct_all_q_planes( self, progress_cb=None, return_base=False, debug_obj=None ):

        self.logger.info("correct_all_q_planes with optins: %s", str((self.baseline_opt, self.baseline_type)))

        if return_base:
            e_base_list = []

        if self.baseline_opt == 0:
            self.logger.info( 'no baseline correction will be performed.' )

        self.lpm_params = []
        for i in range( len(self.qvector) ):

            if return_base:
                base = self.correct_a_single_q_plane( i, return_baseline=True )
                e_base_list.append( base )
                if debug_obj is not None:
                    debug_obj.plot(i, base)
            else:
                try:
                    plot_closure = self.correct_a_single_q_plane( i )
                    if return_base:
                        e_base_list.append( plot_closure )
                except:
                    # 
                    plot_closure = None

                if plot_closure is not None:
                    figsize = ( 16, 8 ) if is_low_resolution() else ( 16, 8 )
                    dc = DebugCanvas( "Debug Z, Zs", plot_closure, figsize=figsize, parent=self.parent )
                    dc.show()
                    if not dc.continue_:
                        break

            # intensity_array[:,i,1] -= baseline
            if progress_cb is not None:
                progress_cb( i )

        self.logger.info( 'correct_all_q_planes done with p_finals %.3g-%.3g.', self.lpm_params[0][1], self.lpm_params[-1][1])
        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            lpm_params = np.array(self.lpm_params)
            plt.push()
            fig, ax = plt.subplots()
            ax.set_title("Offset Percentile Values used with %.3g sigma width" % self.size_sigma, fontsize=16)
            ax.plot(self.qvector, lpm_params[:,0]*100, label='noise level')
            ax.plot(self.qvector, lpm_params[:,1], label='offset percentile')
            ax.legend()
            fig.tight_layout()
            plt.show()
            plt.pop()

        self.lpm_params = None
        if return_base:
            return np.array( e_base_list )

    def correct_a_single_q_plane( self, i, suppress_update=False, plot_always=False, return_baseline=False ):
        if len(self.intensity_array.shape) == 3:
            Z_  = self.intensity_array[:,i,1]
        else:
            Z_  = self.intensity_array[:,i]
        if suppress_update:
            Z   = copy.deepcopy(Z_)
        else:
            Z   = Z_
        Z_orig  = copy.deepcopy(Z)

        spline  = LSQUnivariateSpline( self.jvector, Z_orig, self.iknots )
        noisiness = self.compute_noisiness(spline, Z_orig)
        p_final =  base_percentile_offset( noisiness, size_sigma=self.size_sigma )
        if self.lpm_params is not None:
            self.lpm_params.append((noisiness, p_final))
        if self.log_xray_lpm_params:
            self.logger.info('[%d] noisiness=%g p_final=%g' % (i, noisiness, p_final) )

        baseline_total  = self.compute_baseline_and_corrrect( i, Z, p_final )
        plot_closure    = self.do_adjustment( i, Z, Z_orig, spline, baseline_total, plot_always, return_baseline, p_final )
        return plot_closure

    def compute_noisiness(self, spline, Z):
        pp  = np.percentile( Z, [95, 5] )
        height = pp[0] - pp[1]
        # height = pp[0]*0.5
        noisiness = np.std( Z - spline(self.jvector) ) / height
        return noisiness

    def compute_baseline_and_corrrect( self, i, Z, p_final, noisiness=None ):

        if self.baseline_opt == 0:
            baseline_total  = np.zeros(len(Z))
            return baseline_total

        q   = self.qvector[i]
        j   = self.jvector

        if q < VERIY_SMALL_ANGLE_LIMIT:
            iterate = self.num_iterations * 2
        elif q < SMALL_ANGLE_LIMIT:
            iterate = self.num_iterations
        else:
            iterate = 1

        if i < 10:
            print( 'correcting', [i] )

        try:
            baseline_total = compute_baseline_using_LPM_impl(self.baseline_type, iterate, j, Z, p_final=p_final, k=i, curve=self.curve)

        except Exception as exc:
            raise RuntimeError( 'failed to solve baseline at Q[%d]=%g; %s' % (i, q, str(exc)) )
            # note that intensity_array correnction is not complete in this cas

        return baseline_total

    def do_adjustment( self, i, Z, Z_orig, spline, baseline_total, plot_always, return_baseline, p_final ):
        show_plot   = self.show_plot

        if self.need_adjustment or show_plot:
            # construct Affine trasnformation
            x_vector        = self.x_vector
            x_base          = self.x_base
            x_adjust        = self.x_adjust
            x_total_adjust  = x_base + x_adjust
            x_end_i         = len(x_base)-1
            affine_src_x    = [ 0, self.top_x, x_end_i ]
            affine_src_y    = [ x_base[0], self.top_y, x_base[-1] ]
            src_points      = list( zip( affine_src_x, affine_src_y ) )

            affine_tgt_x    = affine_src_x
            tgt_hpm_y       = spline(self.hpm_i)
            tgt_top_y       = np.average( tgt_hpm_y )
            affine_tgt_y    = [ baseline_total[0], tgt_top_y, baseline_total[-1] ]
            tgt_points      = list( zip( affine_tgt_x, affine_tgt_y ) )

        if self.need_adjustment:
            affine          = Affine( src_points, tgt_points )

            # affine transform only the end points
            x_adjust_ends   = x_total_adjust[ [0, -1] ]
            tx, ty          = affine.transform( self.jvector[[0, -1]], x_adjust_ends )

            # compute the linear adjustment vector in the target plane
            affined_line    = np.linspace( ty[0], ty[-1], len(x_base) )
            total_line      = np.linspace( baseline_total[0], baseline_total[-1], len(x_base) )
            affined_adjust  = affined_line - total_line

            Z -= affined_adjust     # note that this is updating intensity_array[:,i,1]
        else:
            affined_adjust  = 0

        if return_baseline:
            return baseline_total + affined_adjust

        if plot_always or show_plot and i%10==0:

            # compute the adjusted baseline in the target plane
            affined_total_adjust    = baseline_total + affined_adjust
            src_adjust_points   = [ (0, x_base[0]), (0, x_total_adjust[0]), (x_end_i, x_total_adjust[x_end_i]), (x_end_i, x_base[x_end_i]) ]
            tgt_adjust_points   = [ (0, baseline_total[0]), (0, affined_total_adjust[0]), (x_end_i, affined_total_adjust[x_end_i]), (x_end_i, baseline_total[x_end_i]) ]

            def plot_closure(fig, axes=None, same_scaling=False, zero_base=False, j0=0):
                zero_base = ENABLE_ZERO_BASE and zero_base

                if axes is None:
                    ax1 = fig.add_subplot( 121 )
                    ax2 = fig.add_subplot( 122 )
                else:
                    ax1, ax2 = axes

                total_adjust_color      = 'red'
                adjust_polygon_coloer   = 'pink'

                intensity_picking = get_xray_picking()
                ax1.set_title( 'Elution around Q=%g (averaged)' % ( intensity_picking ) )
                ax2.set_title( 'Elution at Q[%d]=%.3g' % ( i, self.qvector[i] ) )

                x = j0 + np.arange(len(Z))
                affine_src_x_ = list(j0 + np.array(affine_src_x))
                affine_tgt_x_ = list(j0 + np.array(affine_tgt_x))

                if zero_base:
                    zero_base_shift = x_total_adjust
                    w = self.top_y / x_end_i
                    src_peak_shift  = zero_base_shift[0] * ( 1 - w ) + zero_base_shift[x_end_i] * w
                    affine_src_y_   = [ affine_src_y[0] - zero_base_shift[0], self.top_y - src_peak_shift, affine_src_y[2] - zero_base_shift[x_end_i] ]
                    src_points_     = list( zip( affine_src_x_, affine_src_y_ ) )
                    src_adjust_points_  = [ (j0, x_base[0] -  zero_base_shift[0]), (j0, 0), (j0+x_end_i, 0), (j0+x_end_i, x_base[x_end_i] - zero_base_shift[x_end_i]) ]
                else:
                    zero_base_shift = 0
                    affine_src_y_   = affine_src_y
                    src_points_     = list( zip( affine_src_x_, affine_src_y_ ) )
                    # src_adjust_points_ = src_adjust_points
                    src_adjust_points_  = [ (j0, x_base[0]), (j0, x_total_adjust[0]), (j0+x_end_i, x_total_adjust[x_end_i]), (j0+x_end_i, x_base[x_end_i]) ]

                ax1.plot( x_vector - zero_base_shift, color='orange' )

                if self.need_adjustment:
                    ax1.plot( x, x_base - zero_base_shift, color='pink' )
                    ax1.plot( x, x_total_adjust - zero_base_shift, color=total_adjust_color, alpha=0.5 )
                else:
                    ax1.plot( x, x_base - zero_base_shift, color='red' )

                affine_point_colors = [ 'yellow', 'red', 'cyan' ]

                for k in range( 3 ):
                    ax1.plot( affine_src_x_[k], affine_src_y_[k], 'o', color=affine_point_colors[k], markersize=10 )

                src_polygon = Polygon( src_points_, alpha=0.2 )
                ax1.add_patch(src_polygon)

                if self.need_adjustment:
                    src_adjust_polygon = Polygon( src_adjust_points_, alpha=0.3, fc=adjust_polygon_coloer )
                    ax1.add_patch(src_adjust_polygon)

                def draw_text_bpo(ax, bpo):
                    pass

                if zero_base:
                    zero_base_shift = affined_total_adjust
                    w = self.top_y / x_end_i
                    tgt_peak_shift  = zero_base_shift[0] * ( 1 - w ) + zero_base_shift[x_end_i] * w
                    affine_tgt_y_   = [ affine_tgt_y[0] - zero_base_shift[0], tgt_top_y - tgt_peak_shift, affine_tgt_y[2] - zero_base_shift[x_end_i] ]
                    tgt_points_     = list( zip( affine_tgt_x_, affine_tgt_y_ ) )
                    tgt_adjust_points_  = [ (j0, baseline_total[0] - zero_base_shift[0]), (j0, 0), (j0+x_end_i, 0), (j0+x_end_i, baseline_total[x_end_i] - zero_base_shift[x_end_i]) ]
                else:
                    zero_base_shift = 0
                    affine_tgt_y_   = affine_tgt_y
                    # tgt_points_     = tgt_points
                    tgt_points_ = list( zip( affine_tgt_x_, affine_tgt_y ) )
                    # tgt_adjust_points_  = tgt_adjust_points
                    tgt_adjust_points_ = [ (j0, baseline_total[0]), (j0, affined_total_adjust[0]), (j0+x_end_i, affined_total_adjust[x_end_i]), (j0+x_end_i, baseline_total[x_end_i]) ]

                ax2.plot( x, Z_orig - zero_base_shift, color='orange' )
                ax2.plot( x, spline( self.jvector ) - zero_base_shift, ':', color='green' )
                if self.need_adjustment:
                    ax2.plot( x, baseline_total - zero_base_shift, color='pink' )
                    ax2.plot( x, affined_total_adjust - zero_base_shift, color=total_adjust_color, alpha=0.5 )
                else:
                    ax2.plot( x, baseline_total - zero_base_shift, color='red' )

                for k in range( 3 ):
                    ax2.plot( affine_tgt_x_[k], affine_tgt_y_[k], 'o', color=affine_point_colors[k], markersize=10 )

                tgt_polygon = Polygon( tgt_points_, alpha=0.2 )
                ax2.add_patch(tgt_polygon)

                if self.need_adjustment:
                    tgt_adjust_polygon = Polygon( tgt_adjust_points_, alpha=0.3, fc=adjust_polygon_coloer )
                    ax2.add_patch(tgt_adjust_polygon)

                if True:
                    if zero_base:
                        src_hpm_y_  = self.hpm_y - x_total_adjust[self.hpm_i]
                        tgt_hpm_y_  = tgt_hpm_y - affined_total_adjust[self.hpm_i]
                    else:
                        src_hpm_y_  = self.hpm_y
                        tgt_hpm_y_  = tgt_hpm_y

                    ax1.plot( j0+np.array(self.hpm_i), src_hpm_y_, 'o', color='red', markersize=3, alpha=0.5 )
                    ax2.plot( j0+np.array(self.hpm_i), tgt_hpm_y_, 'o', color='red', markersize=3, alpha=0.5 )

                # ymin, ymax = 

                if same_scaling:
                    ymin1, ymax1 = ax1.get_ylim()
                    ymin2, ymax2 = ax2.get_ylim()
                    ymin_ = min( ymin1, ymin2 )
                    ymax_ = max( ymax1, ymax2 )
                    for ax in [ax1, ax2]:
                        ax.set_ylim( ymin_, ymax_ )

                fig.tight_layout()

            ret = plot_closure
        else:
            ret = None

        return ret
