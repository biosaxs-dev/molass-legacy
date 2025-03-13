# coding: utf-8
"""
    ODR_WLS.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
from scipy                  import stats
from scipy.odr              import Model, Data, ODR
from LinearModel            import linear_model

class ODR_WLS_Result:
    def __init__( self, A, B, sigmaA, sigmaB ):
        self.params = np.array( [ A, B ] )
        self.std_errors = np.sqrt( [ sigmaA, sigmaB ] )
        self.sigmaA2    = sigmaA**2
        self.sigmaB2    = sigmaB**2

    def cov_params( self ):
        # this is diagonal only for proof purpose
        return np.diag( [ self.sigmaA2, self.sigmaB2 ] )

class ODR_WLS:
    def __init__( self, y, x, w ):

        slope, intercept, r_value, p_value, std_err = stats.linregress( x, y )

        linear      = Model( linear_model )
        mydata      = Data( x, y, w )
        myodr       = ODR( mydata, linear, beta0=[ slope, intercept ] )
        myoutput    = myodr.run()

        self.B,      self.A         = myoutput.beta
        self.sigmaB, self.sigmaA    = myoutput.sd_beta

    def fit( self ):
        return ODR_WLS_Result( self.A, self.B, self.sigmaA, self.sigmaB )
