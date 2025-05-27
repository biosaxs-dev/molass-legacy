"""
    Optimizer.Devel.TwinPeakMaker.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np

def make_twin_callback_txt(params_type, i, in_file, out_file):
    from molass_legacy.Optimizer.StateSequence import read_callback_txt_impl, save_opt_params
    fv_list, x_list = read_callback_txt_impl(in_file)
    fh = open(out_file, 'w')
    for j, (fv_rec, params) in enumerate(zip(fv_list, x_list)):
        # print([j], fv_rec, params)
        counter, fv, accept, time = fv_rec
        xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), sdmcol_params = params_type.split_params_simple(params)
        xr_v = xr_params[i]/2
        new_xr_params = np.concatenate([xr_params[:i], np.ones(2)*xr_v, xr_params[i+1:]])

        new_rg_params = np.concatenate([rg_params[:i], np.ones(2)*rg_params[i], rg_params[i+1:]])

        uv_v = uv_params[i]/2
        new_uv_params = np.concatenate([uv_params[:i], np.ones(2)*uv_v, uv_params[i+1:]])
        new_params = np.concatenate([new_xr_params, xr_baseparams, new_rg_params, (a,b), new_uv_params, uv_baseparams, (c, d), sdmcol_params])

        save_opt_params(fh, new_params, fv, accept, counter)
    fh.close()

if __name__ == '__main__':
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(this_dir)))
    from ModelParams.SdmParams import SdmParams
    in_file = r"D:\TODO\20240917\C1015F-5\reports\analysis-004\optimized\jobs\063\callback.txt"
    out_file = r"D:\TODO\20240917\C1015F-5\reports\analysis-004\optimized\jobs\064\callback.txt"
    params_type = SdmParams(6)
    make_twin_callback_txt(params_type, 1, in_file, out_file)