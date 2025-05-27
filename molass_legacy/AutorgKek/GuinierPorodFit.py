"""

    GuinierPorodFit.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF

"""
import numpy            as np
from bisect             import bisect_right
from scipy              import optimize

USE_GUINIER_POROD_GENERAL = False
if USE_GUINIER_POROD_GENERAL:
    from DataModels     import GuinierPorodGeneral, GuinierPorodGeneralLmfit
else:
    from DataModels     import GuinierPorod, GuinierPorodLmfit

from lmfit              import minimize, Parameters, Parameter, report_fit
from IntensityData      import ACCEPTABLE_BASIC_QUALITY
# from molass_legacy.KekLib.NumpyUtils         import np_savetxt

DEBUG = False

class GuinierPorodFit:
    def __init__( self, intensity ):
        self.intensity  = intensity

        # if intensity.basically_ok:
        if DEBUG: print( 'basic_quality=', intensity.basic_quality )
        if intensity.basic_quality >= ACCEPTABLE_BASIC_QUALITY:
            f1, t1 = intensity.get_usable_range()
            usable_proportion = 1
        else:
            f1, t1 = intensity.get_approximate_range()
            usable_proportion = 3/4

        f1 = min( 40, f1 )
        if DEBUG: print( 'f1, t1=', [f1,t1] )

        t_ = int( intensity.Q.shape[0] * usable_proportion )

        self.Q = intensity.Q[0:t_]
        self.I = intensity.I[0:t_]
        self.X = intensity.X[0:t_]
        self.Y = intensity.Y[0:t_]
        self.W = intensity.W[0:t_]
        self.E = intensity.E[0:t_]

        self.smoother = intensity.smoother
        self.I_ = self.smoother.predict( self.Q )
        # np_savetxt( 'I_.csv', self.I_ )

        self.model = None
        self.f0 = f1
        self.t0 = t1

    def __call__( self, x, sigma=0.01 ):
        if self.model is None:
            return x * np.nan
        else:
            return self.model( x )

    def fit( self ):

        init_params = self.estimate_init_params()
        # print( 'init_params=', init_params )

        if USE_GUINIER_POROD_GENERAL:
            guinier_porod = GuinierPorodGeneralLmfit()
        else:
            guinier_porod = GuinierPorodLmfit()

        G, Rg, d = init_params

        if USE_GUINIER_POROD_GENERAL:

            min_recchi = None
            opt_result = None

            for s_init in range(3):
                d_init  = max(d, s_init+1)
                params = Parameters()
                params.add('G',     value= G,       min=0 )
                params.add('Rg',    value= Rg,      min=1.0, max=500.0 )
                params.add('d',     value= d_init,  min=1,  max=5 )
                params.add('s',     value= s_init,  min=0,  max=2 )

                result = minimize( guinier_porod, params, args=( self.Q, self.I_, self.W ) )
                G_, Rg_, d_, s_ = result.params['G'].value, result.params['Rg'].value, result.params['d'].value, result.params['s'].value

                if True:
                    print( 'G_, Rg_, d_, s_=', G_, Rg_, d_, s_ )
                    print( 'redchi=', result.redchi )

                if min_recchi is None or result.redchi < min_recchi:
                    min_recchi  = result.redchi
                    opt_result  = result

            G_, Rg_, d_, s_ = opt_result.params['G'].value, opt_result.params['Rg'].value, opt_result.params['d'].value, opt_result.params['s'].value

            if True:
                print( 'G_, Rg_, d_, s_=', G_, Rg_, d_, s_ )
                print( 'redchi=', opt_result.redchi )

            self.result = opt_result
            self.model  = GuinierPorodGeneral( G_, Rg_, d_, s_ )

        else:

            params = Parameters()
            params.add('G',     value= G,   min=0 )
            params.add('Rg',    value= Rg,  min=1.0, max=500.0 )
            params.add('d',     value= d,   min=1,  max=5 )

            result = minimize( guinier_porod, params, args=( self.Q, self.I_, self.W ) )
            G_, Rg_, d_ = result.params['G'].value, result.params['Rg'].value, result.params['d'].value

            self.result = result
            self.model  = GuinierPorod( G_, Rg_, d_ )

        self.I0     = G_
        self.Rg     = Rg_
        self.d      = d_
        self.degree = d_

        # TODO: refactoring
        self.eps    = result.residual
        self.q1     = 1/Rg_ * np.sqrt( 3*d_/2 )
        self.t1     = bisect_right( self.Q, self.q1 )

        self.cover_ratio = 1.0      # TODO: revise it
        cover_  = self.cover_ratio * 100
        self.remarks = 'Guinier-Porod Fit Info: Rg=%.3g, q1=%.3g, q1*Rg=%.3g, d=%.3g, interval=[%d, %d], cover=%d%%' % (
                        self.Rg, self.q1, self.Rg*self.q1, self.degree, self.f0, self.t0, cover_ )


    def equations( self, p ):
        G, Rg, d = p

        if not ( d>=1 and d <= 5 and d>=1 and d <= 5 ):
            return ( np.inf, ) * 6

        if G < 0:
            return ( np.inf, ) * 6

        if Rg <= 0:
            return ( np.inf, ) * 6

        qq  = self.Q_array[0]**2
        f0  = G * np.exp( - qq * Rg**2 / 3 ) - self.I_array[0]

        qq  = self.Q_array[1]**2
        f1  = G * np.exp( - qq * Rg**2 / 3 ) - self.I_array[1]

        q   = np.sqrt( self.Q_array[2] )
        D   = G * np.exp( -d/2 ) * np.power( 3*d/2, d/2 ) / np.power( Rg, d )
        f2  = D / q**d + D / q**d - self.I_array[2]

        return ( f0, f1, f2 )
        # determine G, Rg, d so that this returns ( 0, 0, 0 ) 

    def estimate_init_params( self ):

        # print( 't0=', self.t0 )

        q0  = self.Q[0]
        q1  = self.Q[40]        # TODO: determine a safe value for 20
        t_  = min( 80, self.t0 )
        q2  = self.Q[ t_ ]

        Q_ = np.array( [ q0,  q1,  q2 ] )
        self.Q_array = Q_
        self.I_array = self.smoother.predict( Q_ )

        def f(p):
            return np.abs( np.sum( np.array( self.equations( p ) )**2 ) - 0 )

        G_init = self.I_array[0] / np.exp( -q0**2 * 30.0**2/3 )
        p_init = optimize.fmin( f, ( G_init, 30.0, 2.5 ), disp=False )

        """
        # it suffices to optimize.fmin and no need to optimize.fsolve
        G_, Rg_, d_ = optimize.fsolve( self.equations, p_init )
        print( 'estimate_init_params: ', G_, Rg_, d_ )
        """

        return p_init
