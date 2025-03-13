"""
    Simulative.SeveralStudies.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

def several_studies_impl(lrf_src, guess_info, parent):
    print("several_studies_impl")
    ret_params = lrf_src.draw()     # this is currently equivalent to onthfly_test in StcEstimator.py
    print("ret_params =", ret_params)