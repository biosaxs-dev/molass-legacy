"""
    DriftAnalyzer.py

    Copyright (c) 2018-2023, Masatsuyo Takahashi, KEK-PF
"""
import copy
import numpy as np
from scipy                      import stats
from IntensityData              import IntensityData
from GuinierPorodFit            import GuinierPorodFit
from molass_legacy.Mapping.MapperConstructor import create_mapper
from ScatteringBaseCorrector    import ScatteringBaseCorrector
from DriftLinearModel           import DriftLinearModel

USE_GUINIER_INTERVAL_RATIO  = False
GUINIER_FIT_START_RATIO     = 0.005
GUINIER_FIT_STOP_RATIO      = 0.1

def create_mapper_for_drift_analysis( parent, sd, opt_params=None ):

    if opt_params is None:
        opt_params = {}
        opt_params[ 'uv_baseline_opt' ]         = 1
        opt_params[ 'uv_baseline_type' ]        = 1
        opt_params[ 'uv_baseline_adjust' ]      = 0
        opt_params[ 'xray_baseline_opt' ]       = 1
        opt_params[ 'xray_baseline_type' ]      = 1
        opt_params[ 'xray_baseline_adjust' ]    = 0
        opt_params[ 'dev_allow_ratio' ]         = 0.5

    mapper = create_mapper( parent, sd, opt_params=opt_params )
    return mapper

def get_baselines( sd, mapper, indeces ):

    data_copy = copy.deepcopy( sd.intensity_array )

    params  = mapper.opt_params

    corrector = ScatteringBaseCorrector(
                            sd.jvector,
                            sd.qvector,
                            data_copy,
                            curve=mapper.sd_xray_curve,
                            affine_info=mapper.get_affine_info(),
                            inty_curve_y=mapper.x_curve.y,
                            baseline_opt=params['xray_baseline_opt'],
                            baseline_degree=params['xray_baseline_const_opt'] + 1,
                            need_adjustment=params['xray_baseline_adjust'] == 1,
                            )
    baseline_list = []
    for i in indeces:
        baseline = corrector.correct_a_single_q_plane( i, return_baseline=True )
        baseline_list.append( baseline )

    return baseline_list

class DriftAnalyzer:
    def __init__( self, sd, mapper ):
        self.sd     = sd
        self.mapper = mapper
        self.ecurve = mapper.x_curve
        self.data   = sd.intensity_array

    def solve( self ):
        sd  = self.sd
        qsize   = self.data.shape[1]
        if USE_GUINIER_INTERVAL_RATIO:
            start   = int( qsize * GUINIER_FIT_START_RATIO )
            stop    = int( qsize * GUINIER_FIT_STOP_RATIO )
            print( 'start, stop=', (start, stop) )
        else:
            start   = 5
            stop    = 100
        indeces = np.arange( start, stop, 1, dtype=int )

        baseline_list = get_baselines( sd, self.mapper, indeces )
        model   = DriftLinearModel( sd.qvector, len(sd.jvector) )
        model.fit( indeces, baseline_list )
        print( 'params=', model.params )

        gp_params = []
        for k, info in enumerate(self.ecurve.peak_info):
            print( k, info )
            p   = int( info[1]+0.5 )
            intensity   = IntensityData( self.data[p,:,:], add_smoother=True )
            gp = GuinierPorodFit( intensity )
            gp.fit()
            print( gp.I0, gp.Rg, gp.d )
            gp_params.append( [ gp.I0, gp.Rg, gp.d ] )

        return model.params, gp_params
