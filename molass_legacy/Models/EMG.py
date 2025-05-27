"""

    Models.EMG.py

    Copyright (c) 2017-2024, SAXS Team, KEK-PF

"""
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.patches     import Polygon
from bisect                 import bisect_right
from scipy.interpolate      import UnivariateSpline
from scipy.optimize import minimize
from molass_legacy.KekLib.Affine import Affine
from molass_legacy.Peaks.ElutionModels import emg, VERY_SMALL_VALUE, LARGE_VALUE
from molass_legacy._MOLASS.SerialSettings import get_setting
from .ElutionModelUtils import x_from_height_ratio_impl
from .ModelUtils import _get_model_name

DEVEL = False
if DEVEL:
    from importlib import reload
    import Models.Tentative
    reload(Models.Tentative)
from .Tentative import Model

# task: these limits should be eventually superseded by TAU_BOUND_RATIO
EMG_TAU_LOWER_LIMIT = -50
EMG_TAU_UPPER_LIMIT = 50

def emg_x_from_height_ratio(alpha, mu, sigma, tau):
    n = 100
    hp = mu + tau
    x = np.linspace(hp-4*sigma, hp+5*sigma, n)      # 5*sigma is required for AhRR 
    y = emg(x, 1, mu, sigma, tau)
    hn = np.argmax(y)
    i = bisect_right(y[0:hn], alpha)
    j = bisect_right(-y[hn:], -alpha)

    if False:
        import molass_legacy.KekLib.DebugPlot as dplt
        print('i, hn, jn+j=', i, hn, hn+j)
        fig, ax = dplt.subplots()
        ax.plot(x, y)
        ax.plot(x[hn], y[hn], 'o', color='red')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        try:
            for k in [i, hn+j]:
                ax.plot([x[k], x[k]], [ymin, ymax], ':', color='gray')
        except:
            pass
        dplt.show()

    def get_x0(k):
        # linear approximation
        if k == 0:
            return x[0]
        if k == len(x):
            return x[-1]

        x1 = x[k-1]
        x2 = x[k]
        y1 = y[k-1]
        y2 = y[k]
        return x1 - (x2-x1)/(y2-y1)*(y1-alpha)

    xi = get_x0(i)
    xj = get_x0(hn+j)
    return xi, xj

def emg_x_demo_plot():
    from molass_legacy.KekLib.OurMatplotlib import get_default_colors

    colors = get_default_colors()

    x = np.linspace( 0, 20, 200 )

    h       = 1
    mu      = 10
    sigma   = 1
    tau_list = [ 0, 0.5, 1, 1.5, 2, 3 ]

    fig = plt.figure( figsize=(10, 6) )
    ax  = fig.add_subplot(111)
    ax.set_title( "Extension of EMG for non-positive values of tau", fontsize=16 )

    for k, tau in enumerate( tau_list[1:] ):
        ax.plot( x, emg( x, h, mu, sigma, tau ), ':', color=colors[k+1], label='tau=%g' % ( tau ) )

    for k, tau in enumerate( tau_list ):
        ax.plot( x, emg( x, h, mu, sigma, -tau ), color=colors[k], label='tau=%g' % ( -tau ) )

    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

def emg_guess( x, y ):
    n   = np.argmax( y )
    h   = y[n]

    """
        https://en.wikipedia.org/wiki/Skewness
    """
    v   = np.sum(y)
    m   = np.sum( x*y )/ v
    m2  = np.sum( (x-m)**2 * y )/ v
    m3  = np.sum( (x-m)**3 * y )/ v
    s   = np.sqrt( m2 )
    sk  = m3 / np.power( m2, 3/2 )

    if sk > VERY_SMALL_VALUE:
        tau     = s * np.power( sk/2, 1/3 )
        mu      = m - tau
        sk_c    = 1 - np.power( sk/2, 2/3 )
        # TODO: sk_c <= 0 for OA_Ald
        sigma   = np.sqrt( s**2 * sk_c ) if sk_c > VERY_SMALL_VALUE else s
    elif sk < -VERY_SMALL_VALUE:
        tau     = -s * np.power( -sk/2, 1/3 )
        mu      = m - tau
        sk_c    = 1 - np.power( -sk/2, 2/3 )
        # TODO: sk_c <= 0
        sigma   = np.sqrt( s**2 * sk_c ) if sk_c > VERY_SMALL_VALUE else s
    else:
        tau     = 0
        mu      = m
        sigma   = s

    if False:
        print( 'm=', m, 'm2=', m2, 'm3=', m3, 's=', s, 'sk=', sk )
        print( 'h=', h, 'mu=', mu, 'sigma=', sigma, 'tau=', tau )

    if False:
        fig = dplt.figure()
        ax  = fig.add_subplot(111)
        ax.set_title( "emg_guess" )
        ax.plot( x, y, label='data' )
        ax.plot( x, emg( x, h, mu, sigma, tau ), label='guess' )
        ax.legend()
        dplt.show()

    return h, mu, sigma, tau

class EMG(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, emg, **kwargs)

    def guess(self, y, x=None, negative=False, **kwargs):
        h, mu, sigma, tau = emg_guess( x, y )
        return np.array([h, mu, sigma, tau])

    def eval(self, params=None, **kwargs):
        ret = self.func(**self.make_funcargs(params, kwargs))
        ret[ np.logical_not( np.isfinite(ret) ) ] = LARGE_VALUE
        # TODO: this is not enough. still raises ValueError("The input contains nan values")
        return ret

    def get_name( self ):
        return _get_model_name( self.name )

    def x_from_height_ratio(self, ecurve, ratio, params):
        # mu, sigma, tau
        return x_from_height_ratio_impl(emg, ecurve, ratio, *params[1:])

    def get_peaktop_xy_bad(self, x, params):
        px = np.sum(params[[1,3]])      # mu, tau
        py = self.func(px, *params)
        return px, py

    def get_params_string(self, params):
        return 'h=%g, mu=%g, sigma=%g, tau=%g' % tuple(params)

    def guess_a_peak_with_prop(self, x, y, prop, debug=True):
        """
            This function is used to guess a modeled peak with a given area ratio.
        """
        h, mu, sigma, tau = emg_guess(x, y)
        total_area = np.sum(y)
        def objective(p, debug=False):
            sigma_, tau_  = p
            cy = emg(x, h, mu, sigma_, tau_)
            if debug:
                import molass_legacy.KekLib.DebugPlot as dplt
                with dplt.Dp():
                    fig, ax = dplt.subplots()
                    ax.set_title('sigma=%g, tau=%g' % (sigma_, tau_))
                    ax.plot(x, y, label='data')
                    ax.plot(x, cy, label='guess')
                    ax.legend()
                    fig.tight_layout()
                    dplt.show()
            area = np.sum(cy)
            return abs(area - total_area*prop)
        objective([sigma, tau], debug=debug)
        ret = minimize(objective, [sigma, tau], method='Nelder-Mead')
        sigma, tau = ret.x
        return np.array([h, mu, sigma, tau])

CONSTR_A_SIGMA_RATIO    = 1.5
CONSTR_TAU_SIGMA_RATIO  = 5

def emga_impl(x, h, mu, sigma, tau, a, raise_=False ):
    y   = emg( x, h, mu, sigma, tau )

    if abs(a) < VERY_SMALL_VALUE/max(VERY_SMALL_VALUE, h):
        return y

    xf  = mu - 5*sigma
    xt  = mu + 5*sigma

    src_points  = [ (xf, 0), (mu, h), (xt, 0) ]
    tgt_points  = [ (xf, 0), (mu + a, h ), (xt, 0) ]

    try:
        affine = Affine( src_points, tgt_points, raise_=raise_ )
        x_, y_ = affine.transform( x, y )
        if len(x_) > 1:
            spline = UnivariateSpline( x_, y_, s=0, ext=3 )
            ret = spline( x )
        else:
            ret = y_
    except:
        ret = np.ones(len(x)) * np.nan

    return ret

def emga(x, h, mu, sigma, tau, a):
    if np.isscalar(x):
        x = np.array([x])
        return emga_impl(x, h, mu, sigma, tau, a)[0]
    else:
        return emga_impl(x, h, mu, sigma, tau, a)

def emg_no_affine(x, h=1.0, mu=0.0, sigma=1.0, tau=1.0, a=0):
    return emg(x, h, mu, sigma, tau)

ALLOWANCE_MIN_EQ_MAX = 1e-12

def emga_guess_impl(self, y, x, negative, **kwargs):
    h, mu, sigma, tau = emg_guess( x, y )
    return np.array([h, mu, sigma, tau, 0])

class EMGA(EMG):
    def __init__(self, **kwargs):
        func = emga if get_setting('enable_affine_tran') else emg_no_affine
        Model.__init__(self, func, **kwargs)

    def guess(self, y, x=None, negative=False, **kwargs):
        return emga_guess_impl( self, y, x, negative, **kwargs )

    def eval(self, params=None, x=None):
        ret = self.func(x, *params)
        ret[ np.logical_not( np.isfinite(ret) ) ] = LARGE_VALUE
        # TODO: this is not enough. still raises ValueError("The input contains nan values")
        return ret

    def get_name( self ):
        return _get_model_name( self.name )

    def get_param_hints( self, pkey ):
        assert pkey == 'tau'
        return EMG_TAU_LOWER_LIMIT, EMG_TAU_UPPER_LIMIT

    def get_def_expr( self, params ):
        return "emga( h=%.3g, mu=%.3g, sigma=%.3g, tau=%.3g, a=%.3g )" % (params['h'], params['mu'], params['sigma'], params['tau'], params['a'])

    def x_from_height_ratio(self, ecurve, ratio, params):
        # mu, sigma, tau, a
        return x_from_height_ratio_impl(emga, ecurve, ratio, *params[1:])

    def get_params_string(self, params):
        return 'h=%g, mu=%g, sigma=%g, tau=%g, a=%g' % tuple(params)
    
    def guess_a_peak_with_prop(self, x, y, prop, debug=True):
        """
            This function is used to guess a modeled peak with a given area ratio,
            which has been generated (after the implementation of the same method of EMG)
            completely by Cody AI.
        """
        h, mu, sigma, tau = emg_guess(x, y)
        total_area = np.sum(y)
        m = np.argmax(y)
        px = x[m]
        py = y[m]
        def objective(p, debug=False):
            h_, mu_, sigma_, tau_, a_ = p
            cy = emga(x, h_, mu_, sigma_, tau_, a_)
            k = np.argmax(cy)
            px_ = x[k]
            py_ = cy[k]
            if debug:
                import molass_legacy.KekLib.DebugPlot as dplt
                with dplt.Dp():
                    fig, ax = dplt.subplots()
                    ax.set_title('sigma=%g, tau=%g, a=%g' % (sigma_, tau_, a_))
                    ax.plot(x, y, label='data')
                    ax.plot(x, cy, label='guess')
                    ax.legend()
                    fig.tight_layout()
                    dplt.show()
            area = np.sum(cy)
            return max(np.log10((area - total_area*prop)**2), 5*np.log10((px_ - px)**2 + (py_ - py)**2))

        objective([h, mu, sigma, tau, 0], debug=debug)
        ret = minimize(objective, [h, mu, sigma, tau, 0], method='Nelder-Mead')
        h, mu, sigma, tau, a = ret.x
        return np.array([h, mu, sigma, tau, a])

def emga_demo_plot(with_xy_map=False):
    x   = np.linspace( 0, 300, 300 )
    if with_xy_map:
        fig = plt.figure( figsize=(24,6) )
        # print( fig.get_size_inches() )
        ax  = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
    else:
        fig = plt.figure()
        # print( fig.get_size_inches() )
        ax  = fig.add_subplot(111)

    ax.set_title( 'EMGA: EMG with Affine transformation' )
    ax.set_ylim( -0.1, 1.2 )
    # ax.plot( x, emg( x, 1, 150, 30, 30 ) )
    h       = 1
    mu      = 150
    sigma   = 30
    tau     = 10
    a       = 40
    y1  = emga( x, h, mu, sigma, tau, 0 )
    ax.plot( x, y1 )
    y2  = emga( x, h, mu, sigma, tau, a )
    ax.plot( x, y2 )
    xf  = mu - 5*sigma
    xt  = mu + 5*sigma
    src_points = [ (xf, 0), (mu,h), (xt, 0) ]
    tgt_points = [ (xf, 0), (mu+a,h), (xt, 0) ]
    src_polygon = Polygon( src_points, alpha=0.1 )
    ax.add_patch(src_polygon)
    tgt_polygon = Polygon( tgt_points, alpha=0.1, fc='green' )
    ax.add_patch(tgt_polygon)

    colors  = [ 'orange', 'red', 'red', 'yellow' ]
    points  = [ ( xf, 0), (mu, h), (mu+a, h), (xt, 0) ]
    for k, p in enumerate(points):
        ax.plot( p[0], p[1], 'o', color=colors[k] )

    ax.annotate("(μ, h)", xy=(mu, h), xytext=(mu, h + 0.1), ha='center', arrowprops=dict( arrowstyle="-", color='k' ) )
    ax.annotate("(μ + a, h)", xy=(mu+a, h), xytext=(mu+a, h + 0.1), ha='center', arrowprops=dict( arrowstyle="-", color='k' ) )
    ax.annotate("", xy=(mu+a, h), xytext=(mu, h), size=20, arrowprops=dict( arrowstyle="->", linewidth=3, color='k' ))

    ax.annotate( "(μ - 5*σ, 0)", xy=(xf, 0), xytext=(xf, 0.1), ha='center', arrowprops=dict( arrowstyle="-", color='k' ) )
    ax.annotate( "(μ + 5*σ, 0)", xy=(xt, 0), xytext=(xt, 0.1), ha='center', arrowprops=dict( arrowstyle="-", color='k' ) )

    if with_xy_map:
        ax2.set_title( "Mapping of x in the Affine transformation" )
        ax2.set_xlabel( "Elution No." )
        ax2.set_ylabel( "Transformed Elution No." )
        ax3.set_title( "Mapping of y in the Affine transformation" )
        ax3.set_xlabel( "Intensity of EMG" )
        ax3.set_ylabel( "Intensity of EMGA" )
        affine = Affine( src_points, tgt_points )
        x_, y_ = affine.transform( x, y1 )
        ax2.plot( x, x_, 'o', markersize=3 )
        ax3.plot( y1, y_, 'o', markersize=3 )

    fig.tight_layout()
    plt.show()