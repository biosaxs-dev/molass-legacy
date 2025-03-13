"""

    ScatteringBaseUtil.py

    scattering baseline solver

    Copyright (c) 2017-2022, SAXS Team, KEK-PF

"""
import time
from ScatteringBaseCorrector import ScatteringBaseCorrector, SMALL_ANGLE_LIMIT
from molass_legacy._MOLASS.SerialSettings import get_setting

def apply_baseline_correction_impl( jvector, qvector, intensity_array,
                                    mapped_info=None,
                                    basic_lpm=False,    # 
                                    index=None,
                                    ecurve=None,
                                    progress_cb=None,
                                    return_base=False,
                                    parent=None, with_demo=False, logger=None,
                                    debug_obj=None,
                                    debug=True ):
    if logger is not None:
        t0 = time.time()

    opt_params      = mapped_info.opt_params
    affine_info     = mapped_info.affine_info
    baseline_opt    = opt_params['xray_baseline_opt']
    baseline_type   = opt_params['xray_baseline_type']

    awmf1 = opt_params['xray_baseline_with_bpa']        # seems to be not effective. why?
    awmf2 = get_setting('xray_baseline_with_bpa')
    xray_baseline_with_bpa = awmf1 or awmf2

    if baseline_type == 3:
        # for MF
        baseline_opt = 0    # no LPM correction

    need_adjustment = opt_params['xray_baseline_adjust'] == 1

    corrector = ScatteringBaseCorrector( jvector, qvector, intensity_array,
                                    curve=ecurve,
                                    affine_info=affine_info,
                                    inty_curve_y=ecurve.y,
                                    baseline_opt=baseline_opt,
                                    baseline_type=baseline_type,
                                    need_adjustment=need_adjustment,
                                    parent=parent, with_demo=with_demo )
    xray_baseline_type = get_setting('xray_baseline_type')
    if xray_baseline_type == 0:
        logger.info('no baseline correction.')
        return None

    logger.info("baseline correction with basic_lpm=%s and xray_baseline_type=%d", str(basic_lpm), xray_baseline_type)
    if basic_lpm or xray_baseline_type == 1:
        base = corrector.correct_all_q_planes( progress_cb=progress_cb, return_base=return_base, debug_obj=debug_obj )
    else:
        base = corrector.correct_with_matrix_base( mapped_info, progress_cb=progress_cb, return_base=return_base )

    if xray_baseline_with_bpa and False:
        data = intensity_array[:,:,1].T
        if debug:
            from importlib import reload
            import Baseline.LambertBeerTester
            reload(Baseline.LambertBeerTester)
            from molass_legacy.Baseline.LambertBeerTester import test_compute_base_plane
            BP = test_compute_base_plane(data, index, ecurve, denoise=False)
        else:
            from molass_legacy.Baseline.LambertBeer import compute_base_plane
            BP = compute_base_plane(data, index, ecurve, denoise=False, debug=debug)
        base = BP
        intensity_array[:,:,1] -= BP.T

    if logger is not None:
        t = time.time() - t0
        with_bpa = ' with BPA' if xray_baseline_with_bpa else ''
        logger.info('baseline correction%s took %.3g seconds.', with_bpa, t)

    if return_base:
        return base
