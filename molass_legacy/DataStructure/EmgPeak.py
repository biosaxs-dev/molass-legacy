"""

    EmgPeak.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

"""
import numpy as np
from scipy.interpolate import UnivariateSpline
import logging
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.BasicUtils import ordinal_str
from molass_legacy.KekLib.NumpyUtils import moving_average
from molass_legacy.KekLib.OurMatplotlib import get_default_colors
from molass_legacy.Models.ElutionCurveModels import EGH, egh_x_from_height_ratio
from molass_legacy.KekLib.ExceptionTracebacker import log_exception, warnlog_exception
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Elution.CurveUtils import simple_plot

DEVIATION_LIMIT_RATIO   = 0.01
GLOBAL_ADOPTTION_RATIO  = 1.1
FIT_LIMITS_MIN_RATIO    = 0.05  # > 0.029 and < 0.070 for Ald;
                                # > 0.035 and < 0.074 for OA_Ald
                                # > 0.023 and < 0.183 for Open01
LOCAL_IMPROVE_RATIO     = 0.1
WIDEN_ALLOWANCE_RATIO   = 0.1
EXTENDABLE_DEPRESSION_RATIO = 0.3   # < 0.333 for BL-10C/Ald
WIDE_RANGE_RATIO        = 0.02
USE_MODEL_TOP_X_RANGES  = False
EMG_PEAK_SKIP_OK_RATIO  = 0.1
SIGMA_POINT_RATIO = 2

if USE_MODEL_TOP_X_RANGES:
    from ScipyUtils import erfcxinv

def proof_plot( peak, x_, y_, x, y, fy0, fy1, ry, fx, tx, fy2, chisqrs, chisqrs_, opt_label ):
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.ticker    as ticker

    # FuncFormatter can be used as a decorator
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        # print( 'x, pos=', x, pos )
        if pos is not None:
            if  x - int(x) == 0:
                return "%.3g" % x
            else:
                return ""
        else:
            return "%g" % x

    colors = get_default_colors()

    plt.push()
    fig = plt.figure( figsize=(16, 6) )
    ax1 = fig.add_subplot( 121 )
    ax2 = fig.add_subplot( 122 )
    ax1.set_title( "Fitted model lines and the final residual line", fontsize=16 )
    ax2.set_title( "Chisquares (Σr²/n) of the fits", fontsize=16 )

    ax1.set_xlabel( "Elution No.", fontsize=16 )
    ax1.set_ylabel( "Intensity", fontsize=16 )
    ax2.set_xlabel( "Fit Trial Order", fontsize=16 )
    ax2.set_ylabel( "Chisquare", fontsize=16 )

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    ax2.xaxis.set_major_formatter(major_formatter)

    ax1.plot( x_, y_, color='orange' )
    ax1.plot( peak.top_x, peak.top_y, 'o', color='red' )
    for k, fy in enumerate( [ fy0, fy1, fy2 ] ):
        ax1.plot( x, fy, ':', label=ordinal_str(k) + ' fit' )

    ax1.plot( x, ry, ':', label='final %s residual' % opt_label )

    for i in [fx, tx]:
        ax1.plot( i, y_[i-x_[0]], 'o', color='cyan' )

    hy  = peak.top_y / 2
    ax1.plot( [ peak.top_x - peak.hwhmL, peak.top_x + peak.hwhmR ], [ hy, hy ], color='red' )
    ax1.legend(fontsize=12)

    gl_texts = [ 'global', 'local' ]

    for j, cs in enumerate( [chisqrs, chisqrs_] ):
        fmt = '-' if j == 0 else ':'
        ax2.plot( cs, fmt, color='gray', label=gl_texts[j] )
        for k, c in enumerate(cs):
            ax2.plot( k, c, 'o', label='%s-chisqr%d' % (gl_texts[j], k), color=colors[k] )

    ax2.legend(fontsize=12)

    fig.tight_layout()
    plt.show()
    plt.pop()

class EmgPeak:
    def __init__(self, x_size, top_x, top_y, area_prop=None):
        self.logger = logging.getLogger(__name__)
        self.x_size = x_size
        self.top_x  = int( top_x + 0.5 )
        self.top_y  = top_y
        self.area_prop = area_prop      # this will be set later in UnifiedDecompResult

    def shift_copy(self, x0, x_stop):
        # note that this copy is not complete.
        # refactoring desired
        # also, removing lmfit package should be considered
        from copy import deepcopy

        top_x = self.top_x + x0
        epeak_copy = EmgPeak(x_stop, top_x, self.top_y)
        epeak_copy.model = self.model
        epeak_copy.opt_params = deepcopy(self.opt_params)
        try:
            epeak_copy.opt_params[1] += x0
        except:
            assert False
        half_width = SIGMA_POINT_RATIO * epeak_copy.opt_params[2]
        epeak_copy.sigma_points = [ top_x - half_width, top_x + half_width ]

        return epeak_copy

    def estimate_params( self, x_, y_, lower, upper, allow_wider=False, debug=False ):
        widen_allow = int( len(x_) * WIDEN_ALLOWANCE_RATIO )

        slice_ = slice( lower, upper+1 )
        x   = x_[slice_]
        y   = y_[slice_]

        hm  = self.top_y/2

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("estimate_params entry")
                ax.plot( x_, y_ )
                ax.plot( x, y )
                ax.plot( self.top_x, self.top_y, 'o', color='red' )
                ax.plot( x[[0,-1]], [hm, hm], color='red' )
                plt.show()

        try:
            indL = np.where( np.logical_and( x < self.top_x, y > hm ) )[0][0]
            indR = np.where( np.logical_and( x > self.top_x, y > hm ) )[0][-1]
        except:
            raise Exception( 'failed to find indL or indR' )

        # print( 'indL, indR=', indL, indR, x[indL], x[indR] )

        self.indL   = indL
        self.indR   = indR

        self.hwhmL  = self.top_x - x[indL]
        self.hwhmR  = x[indR] - self.top_x
        self.fwhm   = self.hwhmL + self.hwhmR

        model = EGH()   # EGH is the most robust among EGH, EMG, and EMGA.

        def get_wider_xy():
            wf  = max(0, lower-widen_allow)
            wt  = min(len(x_)-1, upper+widen_allow)
            slice_ = slice( wf, wt+1 )
            wx  = x_[slice_]
            wy  = y_[slice_]
            return wx, wy

        eval_interval = [ indL, indR ]
        f, t = 0, len(x)-1
        fy0, chisqr0, chisqr0_, params0  = self.try_fit( model, x, y, f, t, eval_interval, 0, debug=debug)
        fy1, chisqr1, chisqr1_, params1  = self.try_fit( model, x, y, indL, indR, eval_interval, 1, debug=debug )
        opt_got = False
        init_global_ratio   = chisqr0/chisqr0_
        local_improve_ratio = chisqr1_/chisqr0_
        # print( 'init_global_ratio=', init_global_ratio, 'local_improve_ratio=', local_improve_ratio )
        # local_improve_ratio > LOCAL_IMPROVE_RATIO is neccesary for the 1st peak of Feb25
        if init_global_ratio < GLOBAL_ADOPTTION_RATIO and local_improve_ratio > LOCAL_IMPROVE_RATIO  or allow_wider:
            opt_label = '0th'
            wx, wy = get_wider_xy()
            f, t = 0, len(wx)-1
            offset = max(0, lower-widen_allow)
            eval_interval_w = [ _ + offset for _ in eval_interval ]

            chisqrw = None
            for k, wmodel in enumerate([model]):
                try:
                    fy_, chisqrw, chisqrw_, paramsw = self.try_fit( wmodel, wx, wy, f, t, eval_interval_w, 1.5, debug=debug )
                    if debug:
                        print( '------------------------------------- chisqrw=', chisqrw, 'chisqr0=', chisqr0 )
                    opt_got = True
                    break
                except AssertionError as exc:
                    if debug:
                        log_exception(self.logger, "EmgPeak.estimate_params: ")
                    continue

            if chisqrw is not None and chisqrw < chisqr0:
                opt_f_x, opt_t_x = wx[0], wx[-1]
                opt_params = paramsw
                if debug:
                    self.logger.info("opt_params have been normally set from paramsw with mu=%g", opt_params["tR"].value)
            else:
                opt_f_x, opt_t_x = x[0], x[-1]
                opt_params = params1    # ?

        if debug or not opt_got:
            ry  = moving_average( np.abs( fy1 - y ), keepsize=True )
            f, t = self.find_fit_limits( x, y, fy0, fy1, ry, debug=debug )
            fy2, chisqr2, chisqr2_, params2 = self.try_fit( model, x, y, f, t, eval_interval, 2, debug=debug )

            if opt_got:
                ry  = moving_average( np.abs( fy0 - y ), keepsize=True )
            else:
                # this path is necessary for Ald, etc.
                opt_f_x, opt_t_x = x[f], x[t]
                opt_label = '2nd'
                opt_params = params2
                self.logger.warning("opt_params have been covered from paramsw with mu=%g", opt_params[1])

        if debug:
            wx, wy = get_wider_xy()
            proof_plot( self, wx, wy, x, y, fy0, fy1, ry, opt_f_x, opt_t_x, fy2, [ chisqr0, chisqr1, chisqr2 ], [chisqr0_, chisqr1_, chisqr2_ ], opt_label )

        self.flimL  = opt_f_x
        self.flimR  = opt_t_x
        self.model = model
        self.opt_params = opt_params
        self.sigma_points = self.get_sigma_points( SIGMA_POINT_RATIO )
        # assert self.sigma_points[0] < self.top_x_m and self.top_x_m < self.sigma_points[1]

    def get_model_y(self, x):
        return self.model.eval( self.opt_params, x=x )

    def get_model_x_from_ratio(self, ratio):
        params = self.opt_params
        mu = params[1]
        sigma = params[2]
        tau = params[3]
        ret = egh_x_from_height_ratio(ratio, mu, sigma, tau)
        return ret

    def get_sigma_points(self, ratio, max_sigma_tau=False):
        # TODO: verify or refine this
        params = self.opt_params

        try:
            a = params[4]
        except:
            a = 0

        sigma   = params[2]
        tau = params[3]

        if USE_MODEL_TOP_X_RANGES:
            try:
                mu  = params[1]                     # EMG
                if abs(tau) > 0.1:
                    # this doesn't yet seem to be correct
                    model_top_x = mu + erfcxinv(tau/sigma*np.sqrt(2/np.pi))*sigma*np.sqrt(2) - sigma**2/tau + a
                else:
                    model_top_x = mu + a
            except Exception as exc:
                # print(exc)
                raise exc
        else:
            model_top_x = self.top_x

        if max_sigma_tau:
            scale = max(sigma, tau)
        else:
            scale = sigma

        f = int(max(0, model_top_x - scale*ratio))
        t = int(min(self.x_size-1, model_top_x + scale*ratio))

        self.top_x_m = int(model_top_x + 0.5)
        return [ f, t ]

    def __repr__( self ):
        return 'EmgPeak' + str( ( self.flimL, self.top_x, self.flimR ) )

    def get_fit_limits( self ):
        return self.flimL, self.flimR

    def try_fit( self, model, x, y, f, t, eval_interval, did, debug=False ):
        slice_ = slice( f, t+1 )
        x_  = x[slice_]
        y_  = y[slice_]
        params = model.guess( y_, x=x_ )

        fy = None
        for method in ['least_squares']:
            try:
                out = model.fit(y_, params, x=x_, method=method)
                fy  = model.eval( out.params, x=x )
                break
            except:
                # numpy.linalg.LinAlgError: SVD did not converge, etc.
                warnlog_exception(self.logger, "method=%s failed in model.fit" % method)
                continue

        assert fy is not None

        chisqr  = np.sum( (fy - y)**2 ) / len(x)
        eval_slice = slice( eval_interval[0], eval_interval[1]+1 )
        eval_fy = fy[eval_slice]
        eval_y  = y[eval_slice]
        chisqr_ = np.sum( ( eval_fy - eval_y )**2 ) / len(eval_y)

        if debug:
            plt.push()
            # RuntimeWarning: invalid value encountered in double_scalars
            fig, ax = plt.subplots()
            ax.set_title( "try_fit %g" % did )
            ax.plot( x, y )
            ax.plot( x, fy )
            for k in [f, t]:
                ax.plot( x[k], y[k], 'o', color='cyan' )
            plt.show()
            plt.pop()

        return fy, chisqr, chisqr_, out.params

    def find_fit_limits( self, x, y, fy0, fy1, ry, debug=False ):
        # TODO: consider cases with bad S/N
        delta = self.top_y* DEVIATION_LIMIT_RATIO
        w   = np.where( np.logical_and( x < self.top_x, ry > delta ) )[0]
        f   = w[-1] if len(w) > 0 else 0
        w   = np.where( np.logical_and( x > self.top_x, ry > delta ) )[0]
        t   = w[0] if len(w) > 0 else len(x) - 1

        if debug:
            plt.push()
            ax = plt.gca()
            ax.cla()
            ax.set_title("find_fit_limits")
            ax.plot( x, y )
            for k in [f, t]:
                ax.plot( x[k], y[k], 'o', color='yellow' )
            plt.show()
            plt.pop()

        fx  = x[0:self.indL]
        fy  = ry[0:self.indL]
        try:
            if True:
                f_ratio = np.max(fy) / self.top_y
                # print( 'f_ratio=', f_ratio )
                if f_ratio < FIT_LIMITS_MIN_RATIO:
                    # TODO: this test is not stable
                    f   = 0
            else:
                spline = UnivariateSpline( fx, fy )
                fd0 = spline(fx)
                fd1 = spline.derivative(1)(fx)
                fd2 = spline.derivative(2)(fx)
                len_neg = len( np.where( fd2 < 0 )[0] )
                f_ratio = len_neg / len(fx)
                # print( 'f_ratio=', f_ratio, len_neg, len(fx) )
                if f_ratio < EXTENDABLE_DEPRESSION_RATIO:
                    f   = 0
        except:
            fd2 = None

        tx  = x[self.indR+1:]
        ty  = ry[self.indR+1:]
        try:
            if True:
                t_ratio = np.max(ty) / self.top_y
                # print( 't_ratio=', t_ratio )
                if t_ratio < FIT_LIMITS_MIN_RATIO:
                    # TODO: this test is not stable
                    t   = len(x) - 1
            else:
                spline = UnivariateSpline( tx, ty )
                td0 = spline(tx)
                td1 = spline.derivative(1)(tx)
                td2 = spline.derivative(2)(tx)
                len_neg = len( np.where( td2 < 0 )[0] ) 
                t_ratio = len_neg / len(tx)
                # print( 't_ratio=', t_ratio, len_neg, len(tx) )
                if t_ratio < EXTENDABLE_DEPRESSION_RATIO:
                    t   = len(x) - 1
        except:
            td2 = None

        f   = int( min( self.indL, f) )
        t   = int( max( self.indR, t) )

        if debug:
            # self.extend_debug_plot( x, y, fy0, fy1, ry, f, t, fx, fd0, fd1, fd2, tx, td0, td1, td2 )
            self.extend_debug_plot( x, y, fy0, fy1, ry, f, t, fx, tx )

        return f, t

    # def extend_debug_plot( self, x, y, fy0, fy1, ry, f, t, fx, fd0, fd1, fd2, tx, td0, td1, td2 ):
    def extend_debug_plot( self, x, y, fy0, fy1, ry, f, t, fx, tx ):
        plt.push()
        ax  = plt.gca()
        ax.cla()
        ax_ = ax.twinx()
        ax_.cla()
        ax.plot( x, y )
        ax.plot( x, fy0, ':', label='fy0' )
        ax.plot( x, fy1, ':', label='fy1' )
        ax.plot( x, ry )
        ax.plot( self.top_x, self.top_y, 'o', color='red' )
        hm  = self.top_y/2
        ax.plot( x[ [self.indL, self.indR] ], [hm, hm], color='red' )

        for i in [f, t]:
            ax.plot( x[i], y[i], 'o', color='orange' )

        if False:
            labels = [ 'fd0', 'fd1', 'fd2', 'td0', 'td1', 'td2' ]
            for k, point in enumerate( [ [fx, fd0], [fx, fd1], [fx, fd2], [tx, td0], [tx, td1], [tx, td2] ] ):
                dx, dy = point
                if dy is None:
                    continue

                if k in [0, 3]:
                    ax.plot( dx, dy, ':', label=labels[k] )
                else:
                    ax_.plot( dx, dy, ':', label=labels[k] )
                    neg_dy = dy < 0
                    ax_.plot( dx[neg_dy], dy[neg_dy], ':', color='red' )

        ax.legend( loc='upper left' )
        ax_.legend( loc='upper right' )
        plt.show()
        plt.pop()

    def get_assert_info(self):
        return ( self.flimL, self.top_x, self.flimR )

    def get_xkey( self ):
        return '%.3g' % (self.top_x)

    def get_params(self):
        return self.opt_params[0:4]

class EmgPeakLite(EmgPeak):
    def __init__( self, L, M, R):
        self.flimL  = L
        self.top_x  = M
        self.flimR  = R

class EmgPeakProxy(EmgPeak):
    def __init__( self, top_x, top_y, area_prop=None, model=None, opt_params=None):
        self.top_x  = top_x
        self.top_y  = top_y
        self.area_prop = area_prop
        self.model = model
        self.opt_params = opt_params
        sigma = opt_params[2]
        self.flimL = int(top_x - sigma)     # ok?
        self.flimR = int(top_x + sigma)     # ok?
        self.sigma_points = [ top_x - SIGMA_POINT_RATIO*sigma, top_x + SIGMA_POINT_RATIO*sigma ]
        self.sign = 1

def get_peaks( curve, max_y=None, allow_wider=False, orig_y=None, logger=None, debug=False ):

    local_debug = get_setting("local_debug")

    x   = curve.x
    y   = curve.y

    # if debug or len(curve.peak_info) == 0:
    if debug or local_debug:
        import inspect
        for frm in inspect.stack()[1:10]:
            print("get_peaks: %s %s (%d)" % (frm.filename, frm.function, frm.lineno))

        print("(1) curve.peak_info=", curve.peak_info)

        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("get_peaks entry")
            simple_plot(ax, curve)
            plt.show()

    if False:
        try:
            wide_ranges = curve.get_ranges_by_ratio(WIDE_RANGE_RATIO)
            # print( 'wide_ranges=', wide_ranges )
        except:
            log_exception(logger, "curve.get_ranges_by_ratio")

    peaks = []
    for k, info in enumerate(curve.peak_info):
        lower, top, upper = [x[j] for j in info]
        ratio = (top - lower)/(upper - lower)
        if debug or local_debug:
            print([k], (lower, top, upper), ratio)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("%d-th peak plot" % k)
                ax.plot(x, y)
                for px, label in [(lower, 'lower'), (top, 'top'), (upper, 'upper')]:
                    ax.plot(px, curve.spline(px), "o", label=label)
                ax.legend()
                fig.tight_layout()
                plt.show()

        if ratio < EMG_PEAK_SKIP_OK_RATIO:
            # for cases such as the 2nd peak of 20181203
            if logger is not None:
                from molass_legacy.KekLib.BasicUtils import ordinal_str
                logger.warning( '%s peak has been skipped due to ratio=%.3g in get_peaks' % (ordinal_str(k+1), ratio) )
            continue
        if top < lower:
            # fix this bug for the residual curve of May11
            continue

        if max_y is not None:
            if curve.spline(top) < max_y:
                continue

        peak = EmgPeak( len(x), top, curve.spline( top ) )
        try:
            peak.estimate_params( x, y, lower - x[0], upper - x[0], allow_wider=allow_wider, debug=debug)
            if orig_y is None:
                peak.sign = 1
            else:
                peak.sign = 1 if orig_y[peak.top_x] > 0 else -1

            peaks.append( peak )
        except:
            if logger is not None:
                warnlog_exception(logger, "get_peaks: ", n=5)
            continue

        if False:
            plt.push()
            fig, ax = plt.subplots()
            ax.set_title("get_peaks: %dth peak debug" % k)
            ax.plot(x, y, label="data")
            y_ = peak.get_model_y(x)
            ax.plot(x, y_, ':', color="C%d" % (k+1), label="%dth peak" % k)
            ptx = info[1]
            ax.plot(ptx, curve.spline(ptx), "o", color="red")
            ax.legend()
            fig.tight_layout()
            plt.show()
            plt.pop()

    if debug and len(peaks) == 0:
        # this can be a case for pH6
        print( 'peak_info=', curve.peak_info )
        plt.push()
        fig, ax = plt.subplots()
        ax.set_title("get_peaks: No emgpeak!")
        simple_plot(ax, curve)
        fig.tight_layout()
        plt.show()
        plt.pop()
        # assert False

    return peaks
