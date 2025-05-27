"""

    Models.EGH.py

    Copyright (c) 2017-2025, SAXS Team, KEK-PF

"""
import numpy as np
from bisect import bisect_right
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from molass_legacy.KekLib.Affine import Affine
from molass_legacy.Peaks.ElutionModels import egh, VERY_SMALL_VALUE, compute_moments, compute_egh_params
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
EGH_TAU_LOWER_LIMIT = -50
EGH_TAU_UPPER_LIMIT = 50

"""
    This implemetation is based on the following paper
        A hybrid of exponential and gaussian functions as a simple model
        of asymmetric chromatographic peaks
        http://acadine.physics.jmu.edu/group/technical_notes/GC_peak_fitting/X_lan_Jorgenson.pdf
"""

def egh_x_from_height_ratio(alpha, tR, sigma, tau):
    alpha_ = np.log(1/alpha)
    v = -tau*alpha_/2
    w = np.sqrt((8*sigma**2 + tau**2*alpha_)*alpha_)/2
    return tR - (v+w), tR - (v-w)

def egh_params( x, params ):
    H   = params['H']
    tR  = params['tR']
    sigma  = params['sigma']
    tau  = params['tau']
    return egh( x, H, tR, sigma, tau )

ALPHA       = 0.1
LN_ALPHA    = np.log(ALPHA)

class EGH(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, egh, **kwargs)

    def guess(self, y, x=None, negative=False, **kwargs):

        n   = np.argmax( y )
        H   = y[n]
        yf  = H * ALPHA

        iL  = bisect_right( y[0:n], yf )
        A   = x[n] - x[iL]
        iR   = bisect_right( list(reversed(y[n:])), yf )
        B   = x[len(x)-1-iR] - x[n]

        if False:
            print( 'iL=', iL, 'A=', A )
            print( 'iR=', iR, 'B=', B )
            print( 'n=', n, 'x[n]=', x[n] )

        tR  = x[n]
        sigma = np.sqrt( -B*A/(2*LN_ALPHA) )
        tau = -(B - A)/LN_ALPHA

        return np.array([H, tR, sigma, tau])

    def get_name( self ):
        return _get_model_name( self.name )

    def x_from_height_ratio(self, ecurve, ratio, params):
        # mu, sigma, tau
        return x_from_height_ratio_impl(egh, ecurve, ratio, *params[1:])

    def get_peaktop_xy(self, x, params):
        mu = params[1]
        py = self.func(mu, *params)
        return mu, py

    def get_params_string(self, params):
        return 'h=%g, mu=%g, sigma=%g, tau=%g' % tuple(params)

    def guess_a_peak_with_prop(self, x, y, prop):
        """
        This function is used to guess a modeled peak with a given area ratio.
        """
        H, tR, sigma, tau = self.guess(y, x=x)
        total_area = np.sum(y)
        def objective(p):
            sigma_, tau_  = p
            cy = egh(x, H, tR, sigma_, tau_)
            area = np.sum(cy)
            return abs(area - total_area*prop)
        
        ret = minimize(objective, [sigma, tau], method='Nelder-Mead')
        sigma, tau = ret.x
        return np.array([H, tR, sigma, tau])

def egha_impl(x, H, tR, sigma, tau, a, raise_=False):
    y   = egh( x, H, tR, sigma, tau )

    if abs(a) < VERY_SMALL_VALUE/max(VERY_SMALL_VALUE, H):
        return y

    xf  = tR - 5*sigma
    xt  = tR + 5*sigma

    src_points  = [ (xf, 0), (tR, H), (xt, 0) ]
    tgt_points  = [ (xf, 0), (tR + a, H ), (xt, 0) ]

    try:
        affine = Affine( src_points, tgt_points, raise_=raise_ )
        x_, y_ = affine.transform( x, y )

        if len(x_) > 1:
            spline = UnivariateSpline( x_, y_, s=0, ext=3 )
            return spline( x )
        else:
            return y_
    except:
        return np.ones(len(x)) * np.nan

def egha(x, H, tR, sigma, tau, a, raise_=False):
    if np.isscalar(x):
        x = np.array([x])
        return egha_impl(x, H, tR, sigma, tau, a, raise_=raise_)[0]
    else:
        return egha_impl(x, H, tR, sigma, tau, a, raise_=raise_)

def egh_no_affine(x, H=1, tR=0, sigma=1.0, tau=1.0, a=0):
    return egh( x, H, tR, sigma, tau )

class EGHA(EGH):
    def __init__(self, **kwargs):
        func = egha if get_setting('enable_affine_tran') else egh_no_affine
        Model.__init__(self, func, **kwargs)

    def guess(self, y, x=None, negative=False, **kwargs):

        n   = np.argmax( y )
        H   = y[n]
        yf  = H * ALPHA

        iL  = bisect_right( y[0:n], yf )
        A   = x[n] - x[iL]
        iR   = bisect_right( list(reversed(y[n:])), yf )
        B   = x[len(x)-1-iR] - x[n]

        if False:
            print( 'iL=', iL, 'A=', A )
            print( 'iR=', iR, 'B=', B )
            print( 'n=', n, 'x[n]=', x[n] )

        tR  = x[n]
        sigma = np.sqrt( -B*A/(2*LN_ALPHA) )
        tau = -(B - A)/LN_ALPHA

        return np.array([H, tR, sigma, tau, 0])

    def get_name( self ):
        return _get_model_name( self.name )

    def get_param_hints( self, pkey ):
        assert pkey == 'tau'
        return EGH_TAU_LOWER_LIMIT, EGH_TAU_UPPER_LIMIT

    def x_from_height_ratio(self, ecurve, ratio, params):
        # mu, sigma, tau, a
        return x_from_height_ratio_impl(egha, ecurve, ratio, *params[1:])

    def get_params_string(self, params):
        return 'h=%g, tR=%g, sigma=%g, tau=%g, a=%g' % tuple(params)

    def guess_a_peak_with_prop(self, x, y, prop):
        """
        This function is used to guess a peak with a given area ratio,
        which has been generated (after the implementation of the same method of EGH)
        completely by Cody AI.
        """
        H, tR, sigma, tau, a = self.guess(y, x=x)
        total_area = np.sum(y)
        def objective(p):
            sigma_, tau_, a_  = p
            cy = egha(x, H, tR, sigma_, tau_, a_)
            area = np.sum(cy)
            return abs(area - total_area*prop)

        ret = minimize(objective, [sigma, tau, a], method='Nelder-Mead')
        sigma, tau, a = ret.x
        return np.array([H, tR, sigma, tau, a])
    
    def guess_binary_peaks(self, x, y, p1, p2, guess_tau=False, debug=False, plot_info=None):
        scy = np.cumsum(y)
        p1y = scy[-1]*p1
        j = bisect_right(scy, p1y)
        params_list = []
        spline = UnivariateSpline(x, y, s=0, ext=3)
        for x_, y_ in [(x[:j],y[:j]), (x[j:],y[j:])]:
            M = compute_moments(x_, y_)
            if guess_tau:
                # this guess is not so good as expected
                tR, sigma, tau = compute_egh_params([M[0], np.sqrt(M[1]), 0], M)
            else:
                # this seems better for this purpose
                tR, sigma, tau = M[0], np.sqrt(M[1]), 0
            h = spline(tR)
            params_list.append( [h, tR, sigma, tau, 0] )
        params_array = np.array(params_list)

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title('EGHA guess_binary_peaks: p1=%g, p2=%g' % (p1, p2), fontsize=16)
                if plot_info is not None:
                    ax.plot(*plot_info, color='gray', alpha=0.3)
                ax.plot(x, y)
                axt = ax.twinx()
                axt.grid(False)
                axt.plot(x, scy, ':')
                axt.plot(x[j], p1y, 'ro')           
                for params in params_array:
                    cy = egha(x, *params)
                    ax.plot(x, cy, '--')
                    ax.axvline(params[1], color='red')
                fig.tight_layout()
                plt.show()

        return params_array