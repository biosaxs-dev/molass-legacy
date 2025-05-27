"""
    IntensitySmootherAveraging.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""
import numpy as np
from molass_legacy.SerialAnalyzer.ElutionalSmootherUtils import LightIntensityData

class IntensitySmootherAveraging:
    def __init__( self, intensity_array, num_curves_averaged ):
        # print( 'IntensitySmootherAveraging: len(intensity_array)=', len(intensity_array),
        #        ', num_curves_averaged=', num_curves_averaged )
        if type( intensity_array ) == np.ndarray:
            data_array = intensity_array
        else:
            data_array = [ intensity.orig_array  for intensity in intensity_array ]
        self.intensity_matrix   = np.array( data_array )
        self.offset_start       = - ( num_curves_averaged//2 )
        self.offset_stop        = self.offset_start + num_curves_averaged
        # print( 'self.intensity_matrix.shape=', self.intensity_matrix.shape )

    def __call__( self, indeces, return_numpy_ndarray=False ):

        intensity_array = []
        slice_array = []
        for i in indeces:
            start   = max( 0, i + self.offset_start )
            stop    = min( self.intensity_matrix.shape[0], i + self.offset_stop )
            slice_  = slice(start, stop)
            data_qi = np.average( self.intensity_matrix[ slice_, :, 0:2 ], axis=0 )
            # apply error propagation rule to error data
            error2  = self.intensity_matrix[ slice_, :, 2 ]**2
            data_e_ = np.sqrt( np.sum( error2, axis=0 ) ).reshape( ( error2.shape[1], 1) )/( stop - start )
            data = np.hstack( [ data_qi, data_e_ ] )
            if return_numpy_ndarray:
                intensity_array.append( data )
                slice_array.append( slice_ )
            else:
                smoothed_intensity = LightIntensityData( orig_array=data )
                intensity_array.append( smoothed_intensity )

        if return_numpy_ndarray:
            return np.array( intensity_array ), slice_array
        else:
            return intensity_array

class ConcentrationSmootherAveraging:
    def __init__( self, c_vector, num_curves_averaged ):
        self.c_vector       = c_vector
        self.offset_start   = - ( num_curves_averaged//2 )
        self.offset_stop    = self.offset_start + num_curves_averaged

    def __call__( self, indeces ):
        concentration_array = []
        for i in indeces:
            start   = max( 0, i + self.offset_start )
            stop    = min( len( self.c_vector ), i + self.offset_stop )
            concentration_array.append( np.average( self.c_vector[start:stop] ) )

        return np.array( concentration_array )
