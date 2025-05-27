# coding: utf-8
"""
    MicrofluidicMapping.py

    Copyright (c) 2018-2021, SAXS Team, KEK-PF
"""
import copy
import numpy as np
from scipy.interpolate import UnivariateSpline
from lmfit import minimize, Parameters
from molass_legacy.Mapping.SingleComponent import SingleComponent
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.DataStructure.AnalysisRangeInfo import AnalysisRangeInfo

DEBUG = False
if DEBUG:
    import molass_legacy.KekLib.DebugPlot as plt

def make_microfluidic_mapping_info_impl(mapper, opt_params):
    mapper.opt_params = opt_params

    mapper.a_base_adjustment  = 0
    mapper.x_base_adjustment  = 0

    compute_mapping_params(mapper)
    determine_mapping_ranges(mapper)

    mapper.a_curve = mapper.make_a_curve(opt_params)
    mapper.x_curve = mapper.make_x_curve(opt_params)
    mapper.x_curve_y_adjusted = mapper.x_curve.y

    if DEBUG:
        fig = plt.figure(figsize=(12,6))
        fig.suptitle("make_microfluidic_mapping_info_impl")
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(mapper.a_curve.y)
        ax1.plot(mapper.a_spline(mapper.a_curve.x))
        ax2.plot(mapper.x_curve_y_adjusted)
        ax2.plot(mapper.x_curve.y)
        ax2.plot(mapper.x_base, color='red')
        fig.tight_layout()
        plt.show()

    mapper.mapped_vector = make_microfluidic_mapped_vector(mapper)
    mapper.mapped_spline  = UnivariateSpline( mapper.x_x, mapper.mapped_vector, s=0, ext=3 )
    mapper.inv_mapped_boundaries = mapper.x_curve.boundaries

    mapper.in_xray_adjustment_mode = False

    mapper.compute_std_diff()

    mapper.scomp = SingleComponent(mapper)
    mapper.logger.info( 'made microfluidic mapping info' )

def compute_mapping_params(mapper):
    mapper.A_init = A_init = len(mapper.a_vector) / len(mapper.x_vector)
    mapper.B_init = B_init = 0
    S_init = mapper.x_vector[-1] / mapper.a_vector[-1]
    params = Parameters()
    params.add( 'A', value=A_init,      min=A_init-0.5, max=A_init+1.0 )
    params.add( 'B', value=B_init,      min=B_init-50,  max=B_init+50 )
    params.add( 'S', value=S_init,      min=S_init*0.25, max=S_init*4 )

    def objective_func(params):
        A = params['A']
        B = params['B']
        S = params['S']
        i = mapper.x_x
        j = A * i + B
        a_y = mapper.a_spline(j)
        return a_y*S - mapper.x_vector

    result = minimize( objective_func, params, args=(), method='least_squares'  )
    A_opt = result.params['A'].value
    B_opt = result.params['B'].value
    S_opt = result.params['S'].value
    mapper.opt_results = [ [ A_opt, B_opt, S_opt ] ]

def determine_mapping_ranges(mapper):
    upper = 0.9
    ratios = [ 0.1, upper, upper ]
    x_ranges = [[ int(len(mapper.x_vector)*r + 0.5)  for r in ratios ]]
    set_setting('range_type', 5)
    """
        Note that
        setting the same value(upper) and setting range_type to 5 here
        ensures the identification of microfluidic peak in later stages.
        See also AnalysisRangeInfo.upgrade_ranges.
    """

    analysis_range_info = AnalysisRangeInfo(x_ranges)
    set_setting('analysis_range_info', analysis_range_info)

    mapper.x_ranges = x_ranges
    mapper.mapping_ranges = x_ranges
    mapper.peak_eval_ranges = x_ranges
    mapper.uv_peak_eval_ranges = [[ int(len(mapper.a_vector)*r + 0.5)  for r in ratios ]]

def make_microfluidic_mapped_vector(mapper):
    opt_result = mapper.opt_results[0]
    A_opt = opt_result[0]
    B_opt = opt_result[1]
    S_opt = opt_result[2]
    return mapper.a_spline( A_opt * mapper.x_x + B_opt ) * S_opt
