"""
    ElutionModelScaler.py

    Copyright (c) 2018-2023, SAXS Team, KEK-PF
"""
import copy
import numpy as np
from scipy.optimize import minimize
from molass_legacy.KekLib.BasicUtils import Struct, ordinal_str
import molass_legacy.KekLib.DebugPlot as plt

REMOVE_H_RATIO_LIMIT = 0.01     # < 0.0199 for HasA(EGH)

def scale_decomposed_elutions(x, y, mapped_y, mapped_max_y, opt_recs, model, logger=None, debug_info=None):
    """
        mapped_y may be mapped_uv_vector or corrected xray_vector.
        opt_recs may be updated in abnormal cases where it results in a too small amplitude.
    """
    assert len(opt_recs) > 0

    if debug_info is not None:
        from OptRecsUtils import debug_plot_opt_recs
        ecurve, title = debug_info
        debug_plot_opt_recs(ecurve, opt_recs, title=title + ": entry")

    tmp_recs = copy.deepcopy(opt_recs)

    copied_y = copy.deepcopy(mapped_y)

    init_height_params = []
    for j, rec in enumerate(opt_recs):
        k   = rec[0]
        if k < 0:
            # do minor elements later
            init_height_params.append(None)
            continue

        fnc = rec[1]
        top_x = rec[3].top_x
        scale = mapped_y[top_x] / y[top_x]
        height = fnc( top_x ) * scale
        copied_y -= fnc(x) * scale
        init_height_params.append( height )

    for j, rec in enumerate(opt_recs):
        k   = rec[0]
        if k >= 0:
            continue

        fnc = rec[1]
        top_x = rec[3].top_x

        # get minor elements init params from the residual
        scale = copied_y[top_x] / y[top_x]
        height = fnc( top_x ) * scale
        init_height_params[j] = height

    def obj_func(height_params, debug=False):
        if debug:
            print( 'init_height_params=', init_height_params )
            fig = plt.figure()
            ax  = fig.gca()
            ax.set_title( "decompose_uv_elution debug: model=" + model.get_name() )
            ax.plot( x[[0, -1]], [mapped_max_y, mapped_max_y], ':', color='red', label='mapped_max_y' )
            ax.plot( x, mapped_y, color='blue', label='data' )

        try:
            resid_y = copy.deepcopy(mapped_y)
            for k, h in enumerate(height_params):
                rec = tmp_recs[k]
                fnc = rec[1]
                fnc.update_param(0, h)
                fy = fnc(x)
                if debug:
                    ax.plot(x, fy, label='%s' % str(k))
                resid_y -= fy

            if debug:
                ax.plot( x, resid_y, ':', label='residuals' )
                ax.legend()
                fig.tight_layout()
                plt.show()

            error = np.sum( resid_y**2 )
        except:
            error = np.inf

        return error

    result = minimize( obj_func, np.array(init_height_params) )

    if False:
        obj_func(result.x, debug=True)

    ret_recs = []
    to_remove = []
    for k, h in enumerate(result.x):
        rec = tmp_recs[k]
        fnc = rec.evaluator
        fnc.update_param(0, h)
        if rec.peak.sign > 0:
            max_h = np.max(fnc(x))
        else:
            max_h = np.min(fnc(x))
        max_h_ratio = max_h/mapped_max_y
        # print([k], 'max_h_ratio=', max_h_ratio)
        if abs(max_h_ratio) < REMOVE_H_RATIO_LIMIT:
            """
            such as the 3rd element of 20180204/Close01
            """
            to_remove.append([k, max_h_ratio])
            continue

        ret_recs.append(rec)

    for k, ampl_ratio in reversed(to_remove):
        opt_recs.pop(k)
        if logger is not None:
            logger.warning('%s elution element removed due to too small amplitude ratio %.3g' % (ordinal_str(k+1), ampl_ratio) )

    if False:
        obj_func(result.x, debug=True)

    if debug_info is not None:
        from OptRecsUtils import debug_plot_opt_recs
        ecurve, title = debug_info
        debug_plot_opt_recs(ecurve, opt_recs, title=title + ": result")

    return Struct(opt_recs=ret_recs)
