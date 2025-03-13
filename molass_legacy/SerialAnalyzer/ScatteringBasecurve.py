# coding: utf-8
"""

    ScatteringBasecurve.py

    scattering baseline solver

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import numpy            as np
from lmfit              import minimize, Parameters
from scipy              import stats
import OurStatsModels   as sm

DEBUG_PLOT              = False
PERCENTILE_FIRST        = 30
PERCENTILE_SECOND       = 25
PERCENTILE_FINAL        = 20
NEGATIVE_VALUE_PENALTY  = 10

class ScatteringBasecurve:
    def __init__( self, y ):
        self.y  = y
        self.x = np.arange( len(self.y) )

    def solve( self, p_final=PERCENTILE_FINAL ):
        xpp, ypp, A, B, C, ppp = self.get_low_percentile_params( self.x, self.y, p_final )

        b_  = A*self.x**2 + B*self.x + C
        y_  = self.y - b_
        yf  = np.percentile( y_, p_final )
        p_  = np.where( y_ <= yf )[0]
        n   = np.argmax( y_[p_] )
        xf  = p_[n]
        C_  = self.y[xf] - ( A*xf**2 + B*xf )

        self.xpp    = xpp
        self.ypp    = ypp
        self.npp    = n
        self.params = ( A, B, C_ )
        return A, B, C_

    def solve_quadratic( self, xpp, ypp ):
        """
            y = A * X**2 + B * x + C

            X = [   [ 0**2, 0, 1 ],
                    [ 1**2, 1, 1 ],
                    [ 2**2, 2, 1 ],
                    ...
                    [ (n-1)**2, (n-1), 1 ],
                ]
        """
        Xpp = np.array( [ xpp**2, xpp, np.ones( len(xpp) ) ] ).T
        model   = sm.OLS( ypp, Xpp )
        result  = model.fit()
        A, B, C = result.params
        return A, B, C

    def get_low_percentile_params( self, x, y, p_final ):
        ppp = np.percentile( y, [ PERCENTILE_FIRST ] )
        xpp = np.where( y <= ppp[0] )[0]
        ypp = y[xpp]
        n   = 2

        for i in range(n):
            A, B, C = self.solve_quadratic( xpp, ypp )
            y_ = y - ( A*x**2 + B*x + C )
            ppp = np.percentile( y_, [ PERCENTILE_SECOND, p_final ] )
            if i == n - 1:
                break

            xpp = np.where( y_ <= ppp[0] )[0]
            ypp = y[xpp]

        return xpp, ypp, A, B, C, ppp

    def debug_plot( self, title="Debug", parent=None ):
        from DebugCanvas    import DebugCanvas
        A, B, C = self.params

        def debug_plot( fig ):
            ax = fig.add_subplot( 111 )

            ax.set_title( title )
            ax.plot( self.y )
            # ax.plot( self.y, ':', color='green' )
            ax.plot( self.xpp, self.ypp, 'o', color='yellow', label='under 25% points' )


            baseline_ = A*self.x**2 + B*self.x + C
            ax.plot( baseline_, color='red', alpha=0.5, label='low percentile linear' )
            ax.plot( self.npp, self.y[self.npp], 'o', color='red', label='10% point' )

            ax.legend()
            fig.tight_layout()

        dc = DebugCanvas( "Debug", debug_plot, parent=parent, toolbar=True )
        dc.show( cancelable=True )
        return dc.continue_
