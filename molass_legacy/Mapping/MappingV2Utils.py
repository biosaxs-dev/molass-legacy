"""
    MappingV2Utils.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting
from .MappingParams import MappedInfo

def make_mapped_info_for_v2(x_curve):
    # this function separates v2 from the old v1 mapping

    unified_baseline_type = get_setting("unified_baseline_type")
    if unified_baseline_type == 1:
        xray_baseline_type = 1          # linear
    elif unified_baseline_type in [2, 3]:
        xray_baseline_type = 5          # integral
    else:
        assert False

    # affine_info
    # see ElutionMapper.get_affine_info()
    x = x_curve.x
    x_base = np.zeros(len(x))
    x_base_adjustment = 0
    peaks   = [ info[1] for info in x_curve.peak_info ]
    peak_x  = np.average( peaks )
    peak_y  = np.average( x_curve.spline( peaks ) )
    affine_info = [ x, x_base, x_base_adjustment, (peak_x, peak_y), peaks ]

    # see also  ElutionMapper.prepare_env_for_plain_LPM(self):
    opt_params = {  'xray_baseline_opt':1,
                    'xray_baseline_type':xray_baseline_type,
                    'xray_baseline_adjust':0,
                    'xray_baseline_with_bpa':1}

    return MappedInfo(opt_params=opt_params, affine_info=affine_info, x_curve=x_curve, x_base=x_base)
