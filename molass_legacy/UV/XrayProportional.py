"""
    UV/XrayProportional.py

    Copyright (c) 2018-2023, SAXS Team, KEK-PF
"""
import os
import numpy as np
from scipy.interpolate import UnivariateSpline
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Elution.CurveUtils  import get_xray_elution_vector
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy.KekLib.NumpyUtils import np_loadtxt
from ScatteringBaseCorrector import compute_baseline_using_LPM_impl
from molass_legacy.Mapping.SingleComponent import SingleComponent
import molass_legacy.KekLib.DebugPlot as plt

def make_xray_proportinal_uv_data_impl(xray_array, ivector=None):
    this_dir = os.path.dirname( os.path.abspath( __file__ ) )
    uv_ridge_data, _ = np_loadtxt( this_dir + '/UV.dat' )

    lvector_xp     = uv_ridge_data[:, 0]
    uv_ridge_curve  = uv_ridge_data[:, 1]

    if ivector is None:
        qvector = xray_array[0, :, 0]
        ivector, _ = get_xray_elution_vector( qvector, xray_array )

        if get_setting('apply_backsub') == 0:
            ivector_    = ivector
        else:
            ivector_    = copy.deepcopy(ivector)
            compute_baseline_using_LPM_impl( 0, 2, np.arange( len(ivector_) ), ivector_ )
    else:
        ivector_ = ivector

    x_curve = ElutionCurve( ivector_ )
    elution = x_curve.y / x_curve.max_y
    uv_array_xp    = np.dot( uv_ridge_curve.reshape( len(uv_ridge_curve), 1 ), elution.reshape( 1, len(elution) ) )
    uv_file_xp     = 'xray_proportinal.dat'

    return lvector_xp, uv_array_xp, uv_file_xp

def make_proportional_mapping_info_impl(mapper, opt_params, debug=False):
    mapper.opt_params = opt_params

    mapper.a_base_adjustment  = 0
    mapper.x_base_adjustment  = 0
    mapper.A_init = 1
    mapper.B_init = 0
    mapper.a_curve = mapper.make_a_curve(opt_params)
    mapper.x_curve = mapper.make_x_curve(opt_params)
    mapper.logger.info("made proportional curves with len(a_curve.x)=%d, len(x_curve.x)=%d", len(mapper.a_curve.x), len(mapper.x_curve.x))

    scale = np.max(mapper.a_vector) / np.max(mapper.x_vector)

    if len(mapper.x_x) == len(mapper.a_x):
        mapper.a_base = mapper.x_base * scale
    else:
        """
            previously, len(mapper.x_x) and len(mapper.a_x) were the same.
            the situation seems to have been changed probably due to Python (extension modules) updates.
            therefore, adjusting the difference using a spline
        """
        a_base_spline = UnivariateSpline(mapper.x_x, mapper.x_base * scale, s=0)
        mapper.a_base = a_base_spline(mapper.a_x)

    mapper.x_curve_y_adjusted = mapper.x_curve.y

    if debug:
        for v in [mapper.a_x, mapper.x_x, mapper.a_vector, mapper.a_base, mapper.x_base]:
            print(len(v))
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("make_proportional_mapping_info_impl debug")
            ax1.plot(mapper.a_vector)
            ax1.plot(mapper.a_base)

            ax2.plot(mapper.x_curve_y_adjusted)
            ax2.plot(mapper.x_curve.y)
            ax2.plot(mapper.x_base, color='red')
            fig.tight_layout()
            plt.show()

    mapper.determine_mapping_ranges()
    make_proportional_opt_results_impl(mapper)
    mapper.set_peak_eval_ranges()
    mapper.std_diff = 0
    mapper.opt_results

    mapper.mapped_vector = mapper.x_curve_y_adjusted
    mapper.mapped_spline  = UnivariateSpline( mapper.x_x, mapper.mapped_vector, s=0, ext=3 )
    mapper.inv_mapped_boundaries = mapper.x_curve.boundaries

    mapper.in_xray_adjustment_mode = False
    mapper.scomp = SingleComponent(mapper)
    mapper.logger.info("made xray-proportional mapping info with len(a_base)=%d, len(a_vector)=%d", len(mapper.a_base), len(mapper.a_vector))
    assert len(mapper.a_base) == len(mapper.a_vector)

def make_proportional_opt_results_impl(mapper):
    A = 1
    B = 0
    mapper.opt_results = []
    mapper.map_params = (A, B)
    S = 1   # TODO
    mapper.scale_params = [S] * len( mapper.mapping_ranges)
    for i, S in enumerate(mapper.scale_params):
        mapper.opt_results.append( [ A, B, S ] )
