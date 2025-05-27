"""

    ファイル名：   SphericalModel.py

    処理内容：

        ゼロ濃度外挿シミュレーションのプロット

    Copyright (c) 2017-2023, Masatsuyo Takahashi, KEK-PF

"""

import numpy            as np
from scipy.stats        import norm
from bisect             import bisect_right
from molass_legacy.KekLib.NumpyUtils         import np_savetxt

#   feigin-svergun-1987 (1.27)
def f( sR ):
     return 3 * ( np.sin( sR ) - sR * np.cos( sR ) ) / ( sR )**3

class XrayScattering:
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

class UvAbsorbance:
    def __init__( self, num_elution_points, wl_vector, model_curve, scale=1 ):
        self.num_elution_points = num_elution_points
        self.wl_vector = wl_vector
        self.model_curve = model_curve
        self.scale = scale

    def generate_data( self ):
        x   = np.linspace(-7, 7, self.num_elution_points)
        y   = norm.pdf( x, 0, 1 )
        y_  = y / norm.pdf( 0, 0, 1 ) * self.scale
        data_list = [ self.wl_vector ]
        for i in range(len(x)):
            data_list.append( y_[i]*self.model_curve )

        self.data_array = np.array( data_list )

    def save( self, file ):
        fh = open( file, "wb" )
        line = "\t" + "\t".join( [ str(i) for i in range(self.num_elution_points+1) ] ) + '\n'
        fh.write( bytearray( line, 'cp932' ) )  # utf8 or cp932
        for j in range(self.data_array.shape[1]):
            line = "\t".join( [ '%g' % f for f in self.data_array[:,j] ] ) + '\n'
            fh.write( bytearray( line, 'cp932' ) )  # utf8 or cp932
        fh.close()

    def get_elutioncurve( self, lambda_, num_points ):
        j = bisect_right( self.wl_vector, lambda_ )
        indeces = np.linspace( 0, self.num_elution_points-1, num_points, dtype=int )
        return self.data_array[ 1:, j ][indeces]

class SerialXrayScattering:
    def __init__( self, R, q_vector, c_vector, c_scale=1, i_scale=1 ):
        self.R  = R
        self.q_vector   = q_vector
        self.c_vector   = c_vector
        self.c_scale    = c_scale
        self.i_scale    = i_scale

    def generate( self ):
        intensity_list = []
        error_list = []
        x = self.q_vector

        for c in self.c_vector:
            c_ = c * self.c_scale
            model = XrayScattering( c_, self.R )
            noise_level = 1/( 0.01 + c)
            intensity = model( x ) * ( 1 + ( noise_level * 1e-4 + ( x - 0.05 ) **2 )* np.random.normal( 0, noise_level, len(x) ) )
            e = np.ones( len(x) ) * 1e-5 * noise_level
            intensity_list.append( intensity )
            error_list.append( e )

        self.intensity_array = np.array( intensity_list ) * self.i_scale
        self.error_array = np.array( error_list ) * self.i_scale

    def save( self, folder, name='Sample_000.dat' ):
        for n in range(len(self.c_vector)):
            path = folder + '/' + name.replace( '000', '%03d' % (n) )

            fh = open( path, "wb" )
            fh.write( bytearray( '# q intensity error\n', 'cp932' ) )
            fh.close()

            data = np.vstack( [ self.q_vector, self.intensity_array[n,:], self.error_array[n,:] ] )
            np_savetxt( path, data.T, mode="a" )
