"""
    ParamSetUtils.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""

def construct_params_type_from_code(func_code, n):
    if func_code < "F0500":
        from ModelParams.EghParams import construct_egh_params_type
        params_type = construct_egh_params_type(n)
    elif func_code < "F0600":
        from ModelParams.LjEghParams import LjEghParams
        params_type = LjEghParams(n)
    elif func_code < "F0700":
        from ModelParams.FdEmgParams import FdEmgParams
        params_type = FdEmgParams(n)
    elif func_code < "F1000":
        from ModelParams.RtEmgParams import RtEmgParams
        params_type = RtEmgParams(n)
    else:
        from ModelParams.StcParams import construct_stochastic_params_type
        params_type = construct_stochastic_params_type(n)
    return params_type
