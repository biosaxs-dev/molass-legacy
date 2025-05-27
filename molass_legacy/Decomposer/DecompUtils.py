"""
    DecompUtils.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.DataStructure.PeakInfo import PeakInfo
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy.Models.ElutionCurveModels import EGH, EGHA, EMG, EMGA
from molass_legacy.Decomposer.ElutionDecomposer import ElutionDecomposer
from molass_legacy.Decomposer.ElutionModelScaler import scale_decomposed_elutions
from molass_legacy.DataStructure.AnalysisRangeInfo import AnalysisRangeInfo
from UnifiedDecompResult import UnifiedDecompResult
import molass_legacy.KekLib.DebugPlot as plt

NORMALIZED_RESID_LIMIT  = 0.02      # < 0.0476 for Kosugi3a
ALLOW_ONE_SIDE_RESULT   = False     # not allowing seems safe judging from 20230303/HasA
                                    # alternatively, consider making ElutionModelScaler more robust
ADD_PEAK_SCI_LIMIT = 70

def get_elm_info(e_elemnts, opt_recs):
    elm_recs = []
    for k, sel in enumerate(e_elemnts):
        if sel == 1:
            fnc = opt_recs[k][1]
            elm_recs.append( [ k, fnc ] )
    return elm_recs

def make_range_info_impl(opt_recs, control_info, specpanel_list=None, debug=False, parent=None, logger=None, no_elm_recs=False):
    select_matrix   = control_info.select_matrix
    editor_ranges   = control_info.editor_ranges
    top_x_list      = control_info.top_x_list

    if logger is not None:
        specpanel_list_info  = "" if specpanel_list is None else ", len(specpanel_list)=%d" % len(specpanel_list)
        logger.info('make_range_info_impl: editor_ranges=%s%s', str(editor_ranges), specpanel_list_info)

    """
        For debugging, see
            DecompEditorFrame.make_range_info
    """
    if debug:
        print('debug start')
        plt.push()
        fig = plt.figure()
        ax  = fig.gca()
        ax.set_title( 'make_range_info_impl debug' )
        x   = parent.mapper.x_curve.x

    ret_ranges = []
    for k, range_list in enumerate(editor_ranges):
        if specpanel_list is not None and len(specpanel_list) > 0:
            panel = specpanel_list[k]
            if len(range_list) == 0 or panel.ignore.get() == 1:
                continue

        top_x = top_x_list[k]
        if no_elm_recs:
            elm_recs = None
        else:
            e_elemnts = select_matrix[k]
            elm_recs = get_elm_info(e_elemnts, opt_recs)
            if debug:
                print([k], 'e_elemnts=', e_elemnts)
                for rec in elm_recs:
                    e   = rec[0]
                    fnc = rec[1]
                    ax.plot(x, fnc(x), label=str(e))
        ret_ranges.append( [ PeakInfo(k, top_x, elm_recs) ] + range_list  )

    if debug:
        print('debug show', 'ret_ranges=', ret_ranges)
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.pop()

    return ret_ranges

class CorrectedBaseline:
    def __init__(self, sd, mapper):
        x_curve = mapper.x_curve
        self.x_curve = x_curve
        self.x  = x_curve.x

        x_baseline = mapper.x_base + mapper.x_base_adjustment
        self.y  = mapper.x_vector - x_baseline
        self.max_y = self.y[x_curve.max_x]

        # TODO: confirm the necessity below
        if mapper.opt_params['xray_baseline_opt'] == 1:
            sd_copy = sd.get_exec_copy(mapper)      # to prevent given sd from being changed
            sd_copy.apply_baseline_correction( mapper.get_mapped_info() )
            sd = sd_copy

        self.data = sd.intensity_array
        self.sd = sd                    # added for VP-Analysis
        self.a_curve = mapper.a_curve   # added for VP-Analysis

def decompose_elution_impl(x_curve, x, y, model, dual_opt, hints_dict, d_curve, logger, debug):

    decomposer  = ElutionDecomposer(x_curve, x, y, model=model, d_curve=d_curve, retry_valley=True, deeply=True, hints_dict=hints_dict)

    if debug:
        print('len(decomposer.fit_recs)=', len(decomposer.fit_recs))

    if dual_opt:
        from DualOptimzer import get_dual_optimized_info, EmgaOptimzer, EghaOptimzer
        if model.get_name() == 'EMGA':
            optimizer_class = EmgaOptimzer
        else:
            optimizer_class = EghaOptimzer

        opt_recs = get_dual_optimized_info(optimizer_class, x_curve, x, y, decomposer.fit_recs, hints_dict, logger, debug=debug)
    else:
        opt_recs = decomposer.fit_recs

    return decomposer, opt_recs

def decompose_elution_xray_to_uv(corbase_info, mapper, model, a_curve, d_curve=None, dual_opt=True, hints_dict=None, logger=None, debug=False):
    x_curve = corbase_info.x_curve
    x       = corbase_info.x
    y       = corbase_info.y

    decomposer, opt_recs = decompose_elution_impl(x_curve, x, y, model, dual_opt, hints_dict, d_curve, logger, debug)
    # print('xray_to_uv(1): len(opt_recs)=', len(opt_recs))

    max_y = mapper.a_curve.max_y
    debug_info = None
    # debug_info = (x_curve, "decompose_elution_xray_to_uv")
    ret_uv = scale_decomposed_elutions(x, y, a_curve.y, max_y, opt_recs, model, logger, debug_info=debug_info)
    # print('xray_to_uv(2): len(opt_recs)=', len(opt_recs))

    opt_recs_uv = ret_uv.opt_recs
    model_name = model.get_name()

    if debug:
        debug_info = ("xray_to_uv", )
    else:
        debug_info = None

    return UnifiedDecompResult(
                    xray_to_uv=True,
                    x_curve=x_curve, x=x, y=y,
                    opt_recs=opt_recs,
                    max_y_xray = x_curve.max_y,
                    model_name=model_name,
                    decomposer=decomposer,
                    uv_y=a_curve.y,
                    opt_recs_uv=opt_recs_uv,
                    max_y_uv = max_y,
                    debug_info=debug_info,
                    )

def decompose_elution_uv_to_xray(corbase_info, mapper, model, a_curve, d_curve=None, dual_opt=True, hints_dict=None, logger=None, debug=False):
    x       = a_curve.x
    y       = a_curve.y
    decomposer, opt_recs_uv = decompose_elution_impl(a_curve, x, y, model, dual_opt, hints_dict, d_curve, logger, debug)
    # print('uv_to_xray(2): len(opt_recs_uv)=', len(opt_recs_uv))

    x_curve = corbase_info.x_curve
    xray_y  = corbase_info.y
    max_y   = corbase_info.max_y

    debug_info = None
    # debug_info = (a_curve, "decompose_elution_uv_to_xray")
    ret_xray = scale_decomposed_elutions(x, y, xray_y, max_y, opt_recs_uv, model, logger, debug_info=debug_info)
    # print('uv_to_xray(2): len(opt_recs_uv)=', len(opt_recs_uv))

    opt_recs = ret_xray.opt_recs
    model_name = model.get_name()

    if debug:
        debug_info = ("uv_to_xray", )
    else:
        debug_info = None

    return UnifiedDecompResult(
                    xray_to_uv=False,
                    x_curve=x_curve, x=x, y=xray_y,
                    opt_recs=opt_recs,
                    max_y_xray = max_y,
                    model_name=model_name,
                    decomposer=decomposer,
                    uv_y=a_curve.y,
                    opt_recs_uv=opt_recs_uv,
                    max_y_uv = a_curve.max_y,
                    debug_info=debug_info,
                    )

def decompose_elution_better(corbase_info, mapper, model=EMGA(), dual_opt=True, hints_dict=None, logger=None, return_both=False, debug=False):
    assert logger is not None

    mapped_uv_vector = mapper.make_uniformly_scaled_vector(scale=1)
    x_curve = corbase_info.x_curve
    a_curve = ElutionCurve(mapped_uv_vector)
    d_curve = None

    peaks = x_curve.get_emg_peaks()
    if len(peaks) == 1:
        min_sci = mapper.get_min_sci()
        if min_sci < ADD_PEAK_SCI_LIMIT:
            scale = mapper.get_x2a_scale()
            try:
                d_curve = ElutionCurve(np.abs(x_curve.y - a_curve.y/scale))
            except ZeroDivisionError:
                # as in 20180617
                from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                log_exception(logger, "decompose_elution_better: could not make d_curve")
                d_curve = None

    if debug:
        from molass_legacy.Elution.CurveUtils import simple_plot
        scale = mapper.get_x2a_scale()
        m_curve = ElutionCurve(mapped_uv_vector/scale)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("decompose_elution_better entry")
            for c in [x_curve, m_curve, d_curve]:
                if c is not None:
                    simple_plot(ax, c)
            fig.tight_layout()
            plt.show()

    final_result = None
    x2u_result = decompose_elution_xray_to_uv(corbase_info, mapper, model, a_curve, d_curve, dual_opt, hints_dict, logger, debug)
    nresid1 = x2u_result.nresid_uv
    if ALLOW_ONE_SIDE_RESULT:
        if nresid1 < NORMALIZED_RESID_LIMIT and not return_both:
            logger.info('decomposer result has been adopted without comparison: %.3g' % (nresid1))
            final_result = x2u_result

    if final_result is None:
        try:
            u2x_result = decompose_elution_uv_to_xray(corbase_info, mapper, model, a_curve, d_curve, dual_opt, hints_dict, logger, debug)
        except AssertionError:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(logger, "decompose_elution_uv_to_xray failed: ")
            u2x_result = None

        if return_both:
            return x2u_result, u2x_result

        if u2x_result is None:
            final_result = x2u_result
        else:
            nresid2 = u2x_result.nresid_uv
            if nresid2 < nresid1:
                logger.info('decomposer result has been replaced after residual comparison: %.3g, %.3g' % (nresid1, nresid2))
                final_result = u2x_result
            else:
                logger.info('decomposer result has been kept after residual comparison: %.3g, %.3g' % (nresid1, nresid2))
                final_result = x2u_result

    final_result.remove_unwanted_elements()

    if debug:
        from OptRecsUtils import debug_plot_opt_recs
        debug_plot_opt_recs(x_curve, final_result.opt_recs, title="decompose_elution_better result: %s" % final_result.model_name )

    return final_result

def debug_elution_plot(title, model, x, y, opt_recs):
    plt.push()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(title + model.get_name())
    ax.plot(x, y)
    for k, rec in enumerate(opt_recs):
        func = rec[1]
        y_ = func(x)
        ax.plot(x, y_, ':', label='element %d' % (k+1))

    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.pop()

def get_peak2elem_dict(opt_recs):
    ret_dict = {}
    for eno, opt_rec in enumerate(opt_recs):
        pno = opt_rec[0]
        if pno < 0:
            continue

        ret_dict[pno] = eno

    print('opt_recs=', opt_recs)
    print('ret_dict=', ret_dict)
    return ret_dict

def make_default_analysis_range_info(sd, mapper, logger, select_fix=None):
    """
        currently, this is used only in test_870_ExtrapolationSolver.py

        TODO: recovery, where to apply LPM
    """
    corbase_info = CorrectedBaseline(sd, mapper)
    ret = decompose_elution_better(corbase_info, mapper, logger=logger)
    max_y = mapper.x_curve.max_y
    x = mapper.x_curve.x
    ret_info = ret.get_range_edit_info(logger=logger)

    if select_fix is not None:
        if select_fix.num_ranges == 0:
            k = select_fix.k
            update_selection(0, ret.opt_recs, select_matrix, k)
            editor_ranges[k].clear()
        else:
            assert False
        print( 'select_matrix=', select_matrix )

    ret_ranges = make_range_info_impl(ret.opt_recs_uv, ret_info)
    analysis_range_info = AnalysisRangeInfo( ret_ranges )
    return analysis_range_info
