# coding: utf-8
"""
    BiComponentSolver.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF

"""
from bisect                 import bisect_right
import numpy                as np
from scipy                  import optimize
from DataModelsAgg          import _GuinierPorodAgg
from DataModels             import _GuinierPorod

REQUIRED_BASIC_QUALITY  = 0.7

class BiComponentSolver:
    def __init__( self, intensity, no_assert=False ):

        if not no_assert:
            assert( intensity.basic_quality >=REQUIRED_BASIC_QUALITY )

        self.intensity  = intensity
        self.smoother   = intensity.smoother
        self.I0_est     = None
        self.start      = 0         # TODO: remove this

    def solve( self, fit=None ):
        from lmfit import minimize, Parameters

        assert( fit is not None )

        if fit is None:
            G1_init     = self.intensity.SQ[0]
            Rg1_init    = 50.0
            Rg2_init    = 20.0
            d1_init     = 3.0
            d2_init     = 3.0
        else:
            self.I0_est = fit.I0
            G1_init     = fit.I0 * 0.7
            Rg1_init    = fit.Rg * np.sqrt( 2.0 )
            Rg2_init    = Rg1_init / 2
            d1_init     = fit.degree
            d2_init     = d1_init

        Q1 = 1/Rg1_init * np.sqrt( 3*d1_init/2 )
        Q2 = 1/Rg2_init * np.sqrt( 3*d2_init/2 )

        self.prepare_equation_data( Q1, Q2 )

        def f(p):
            return np.abs( np.sum( np.array( self.equations( p ) )**2 ) - 0 )

        min_ee      = None
        min_params  = None

        for ratio in [ 0.5, 0.6, 0.7, 0.8, 0.9 ]:

            G1_init_ = self.I0_est * ratio
            p_init = optimize.fmin( f, ( G1_init_, Rg1_init, Rg2_init, d1_init, d2_init ), disp=False )

            G1, Rg1, Rg2, d1, d2 = optimize.fsolve( self.equations, p_init )
            G2 = self.compute_G2( G1, Rg1, Rg2 )

            Q1 = 1/Rg1 * np.sqrt( 3*d1/2 )
            Q2 = 1/Rg2 * np.sqrt( 3*d2/2 )

            self.prepare_equation_data( Q1, Q2 )
            p_init = optimize.fmin( f, ( G1, Rg1, Rg2, d1, d2 ), disp=False )
            G1, Rg1, Rg2, d1, d2 = optimize.fsolve( self.equations, p_init )
            G2 = self.compute_G2( G1, Rg1, Rg2 )

            if Rg1 < Rg2:
                G1, G2, Rg1, Rg2, d1, d2 = G2, G1, Rg2, Rg1, d2, d1

            params  = Parameters()
            params.add('G1',    value= G1,   min=1e-5, max=self.I0_est )
            params.add('Rg1',   value= Rg1,  min=1.0,  max=500 )
            params.add('Rg2',   value= Rg2,  min=1.0,  max=500 )
            params.add('d1',    value= d1,   min=1.0,  max=5.0 )
            params.add('d2',    value= d2,   min=1.0,  max=5.0 )
            result = minimize( self.lmfit_model, params, args=() )

            G1  = result.params['G1'].value
            Rg1 = result.params['Rg1'].value
            Rg2 = result.params['Rg2'].value
            d1  = result.params['d1'].value
            d2  = result.params['d2'].value
            G2  = self.compute_G2( G1, Rg1, Rg2 )

            opt_params  = [ G1, G2, Rg1, Rg2, d1, d2 ]
            opt_ee      = self.evaluate( *opt_params )

            if min_ee is None or opt_ee < min_ee:
                min_ee      = opt_ee
                min_params  = opt_params

        # print( 'BiComponentSolver.solve: min_params=', min_params[2:4] )
        return [ min_ee, min_params, self.I0_est ]

    def prepare_equation_data( self, Q1, Q2 ):
        self.Q_array = np.array( [
                    0.25 * Q1,
                    0.75 * Q1,
                    0.75 * Q1 + 0.25 * Q2,
                    0.25 * Q1 + 0.75 * Q2,
                    1.4  * Q2,
                    min( 2.0 * Q2, self.intensity.Q[-1]),
                  ] )
        self.I_array = self.smoother( self.Q_array )

    def equations( self, p ):
        G1, Rg1, Rg2, d1, d2 = p

        if not ( d1>=1 and d1 <= 5 and d2>=1 and d2 <= 5 ):
            return ( np.inf, ) * len(p)

        if G1 < 0:
            return ( np.inf, ) * len(p)

        G2 = self.compute_G2( G1, Rg1, Rg2 )
        if G2 < 0:
            return ( np.inf, ) * len(p)

        if Rg1 < Rg2:
            return ( np.inf, ) * len(p)

        qq = self.Q_array[1]**2
        f2 = G1 * np.exp( - qq * Rg1**2/3 ) + G2 * np.exp( - qq * Rg2**2/3 ) - self.I_array[1]

        D1 = G1 * np.exp( -d1/2 ) * np.power( 3*d1/2, d1/2 ) / np.power( Rg1, d1 )
        D2 = G2 * np.exp( -d2/2 ) * np.power( 3*d2/2, d2/2 ) / np.power( Rg2, d2 )

        q  = self.Q_array[2]
        f3 = D1 / q**d1 + G2 * np.exp( - q**2 * Rg2**2 / 3 ) - self.I_array[2]
        q  = self.Q_array[3]
        f4 = D1 / q**d1 + G2 * np.exp( - q**2 * Rg2**2 / 3 ) - self.I_array[3]

        q  = self.Q_array[4]
        f5 = D1 / q**d1 + D2 / q**d2 - self.I_array[4]
        q  = self.Q_array[5]
        f6 = D1 / q**d1 + D2 / q**d2 - self.I_array[5]

        return ( f2, f3, f4, f5, f6 )
        # この戻り値が ( 0, 0, 0, 0, 0 ) になるような G1, Rg1, Rg2, d1, d2
        # を求める。

    def compute_G2( self, G1, Rg1, Rg2 ):
        if self.I0_est is None:
            qq = self.intensity.SQ[0]**2
            G2 = ( self.intensity.SI[0] - G1 * np.exp( - qq * Rg1**2/3 ) ) / np.exp( - qq * Rg2**2/3 )
        else:
            G2 = self.I0_est - G1
        return G2

    def evaluate( self, G1, G2, Rg1, Rg2, d1, d2, debug=False, caller='unknown', eval_max=None ):

        Y_  = np.log( _GuinierPorodAgg( [G1, G2], [Rg1, Rg2], [d1, d2], self.intensity.SQ ) )

        e = self.intensity.SW * ( self.intensity.SY - Y_ )
        ee = np.inner( e, e ) / ( self.intensity.SQ.shape[0] - 3 )

        if ( debug
            and self.debugger_queues is not None
            and ( eval_max is None or ee < eval_max )
            ):
            print( 'evaluate: debugger_queues get' )
            dbcmd = self.debugger_queues[0].get( block=True )
            print( 'evaluate: dbcmd=', dbcmd )
            if dbcmd == 'Stop':
                self.debug_stop = True
                raise Exception( 'Stop' )
            elif dbcmd == 'Step':
                self.debugger_queues[1].put( [ caller, ee, [ G1, G2, Rg1, Rg2, d1, d2 ] ] )
            else:
                assert( False )

        return ee

    def evaluate_gp( self, I0, Rg, d ):
        """
            Be aware that this evaluation is performed on the same interval as
            that of the final solution of the solver.
        """
        Y_ = np.log( _GuinierPorod( I0, Rg, d, self.intensity.SQ ) )
        e = self.intensity.SW * ( self.intensity.SY - Y_ )
        ee = np.inner( e, e ) / ( self.intensity.SQ.shape[0] - 3 )

        return ee

    def lmfit_model( self, params ):
        G1  = params['G1'].value
        Rg1 = params['Rg1'].value
        Rg2 = params['Rg2'].value
        d1  = params['d1'].value
        d2  = params['d2'].value

        G2  = self.compute_G2( G1, Rg1, Rg2 )

        Y_  = np.log( _GuinierPorodAgg( [G1, G2], [Rg1, Rg2], [d1, d2], self.intensity.SQ ) )
        e = self.intensity.SW * ( self.intensity.SY - Y_ )
        return e
