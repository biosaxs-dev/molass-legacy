"""
    SecTheory.EghCompat.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import SecTheory.MonoPore
from importlib import reload
reload(SecTheory.MonoPore)
from SecTheory.MonoPore import estimate_monopore_params, get_modified_params
import SecTheory.T0UpperBound
reload(SecTheory.T0UpperBound)
from SecTheory.T0UpperBound import estimate_t0upper_bound
import SecTheory.MomentAnalysis
reload(SecTheory.MomentAnalysis)

def verify_basic_assumptions(prep_info, nc=5, first_only=False, global_opt=False):

    logger = logging.getLogger(__name__)

    sd = prep_info.sd
    in_folder = prep_info.in_folder
    rg_curve = prep_info.rg_curve

    D, E, qv, ecurve = sd.get_xr_data_separate_ly()

    ecurve = sd.get_xray_curve()
    # estimate_sec_params(ecurve, debug=True)
    # return

    t0_upper_bound = estimate_t0upper_bound(ecurve)
    print("------------------- t0_upper_bound=", t0_upper_bound)
    ret = estimate_monopore_params(ecurve, rg_curve, nc,
                                    t0_upper_bound=t0_upper_bound, global_opt=global_opt, logger=logger, debug=True)
    if first_only:
        return

    modified_params = get_modified_params(ecurve, nc, ret.x, logger=logger, debug=True)
    if modified_params is not None:
        ret = estimate_monopore_params(ecurve, rg_curve, nc,
                                        t0_upper_bound=t0_upper_bound, init_params=modified_params, logger=logger, debug=True)
