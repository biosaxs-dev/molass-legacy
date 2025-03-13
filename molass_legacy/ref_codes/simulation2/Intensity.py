import numpy                as np

#   svergun-1987 (1.27)
def f( sR ):
     return 3 * ( np.sin( sR ) - sR * np.cos( sR ) ) / ( sR )**3

class Intensity:
    def __init__( self, N, R ):
        self.N  = N
        self.R  = R
        self.V  = 4 * np.pi * R**3 / 3
        self.v1 = self.V / N        # volume per particle

    def term1( self, x ):
        sR  = x * self.R
        return self.N * f(sR)**2

    def term2( self, x ):
        sR  = x * self.R
        return 8 * self.V / self.v1 * f( 2*sR )

    def __call__( self, x ):
        return self.term1( x ) * ( 1 - self.term2( x ) )
