"""
    CfsEvalPeaks.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

CFSE_WEIGHT = 5

def get_cfs_evaluated_peaks(x, y, num_peaks, using_hybrid=False, using_cfs=True, debug=False):
    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy.QuickAnalysis.ModeledPeaks import recognize_peaks
    from molass_legacy.Models.Characteristic import CfSpace
    from molass_legacy.Models.ElutionCurveModels import egha, emga
    from molass_legacy.Models.Egh2Emg import to_emga_params
    from SecTheory.Edm import edm_func
    from SecTheory.EdmSpike import guess_edm_initial_params

    if using_hybrid or using_cfs:
        cfs = CfSpace()
        cft0 = cfs.compute_cf(x, y)

    try_num_peaks = num_peaks + 1
    egha_params_list = recognize_peaks(x, y, exact_num_peaks=try_num_peaks, affine=True)

    if debug:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        ax1.plot(x, y)

    shape = (try_num_peaks, 5)

    def objective_egha_cfs(params, debug=False):
        cy_list = []
        for p in params.reshape(shape):
            cy = egha(x, *p)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.plot(x, y)
                for cy in cy_list:
                    ax.plot(x, cy, ":")
                ax.plot(x, ty, ":", color="red")
                fig.tight_layout()
                plt.show()

        if using_hybrid:
            cft1 = cfs.compute_cf(x, ty)
            return np.log(np.sum((ty - y)**2)) + CFSE_WEIGHT*np.log(np.sum((cft1 - cft0)**2))
        elif using_cfs:
            cft1 = cfs.compute_cf(x, ty)
            return np.sum((cft1 - cft0)**2)
        else:
            return np.sum((ty - y)**2)

    params = np.array(egha_params_list).flatten()
    # objective_egha_cfs(params, debug=True)

    ret_egha = minimize(objective_egha_cfs, params)
    # objective_egha_cfs(ret_egha.x, debug=True)

    def objective_emga_cfs(params, debug=False):
        ty = np.zeros(len(x))
        for p in params.reshape(shape):
            cy = emga(x, *p)
            ty += cy

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.plot(x, ty)
                fig.tight_layout()
                plt.show()

        if using_hybrid:
            cft1 = cfs.compute_cf(x, ty)
            return np.log(np.sum((ty - y)**2)) + CFSE_WEIGHT*np.log(np.sum((cft1 - cft0)**2))
        elif using_cfs:
            cft1 = cfs.compute_cf(x, ty)
            return np.sum((cft1 - cft0)**2)
        else:
            return np.sum((ty - y)**2)

    emga_params_list = []
    for params in ret_egha.x.reshape(shape):
        emga_params_list.append(to_emga_params(params))

    params = np.array(emga_params_list).flatten()
    # objective_emga_cfs(params, debug=True)

    ret_emga = minimize(objective_emga_cfs, params)
    # objective_emga_cfs(ret_emga.x, debug=True)

    edm_shape = (try_num_peaks, 6)

    def objective_edm_cfs(params, debug=False):
        ty = np.zeros(len(x))
        for p in params.reshape(edm_shape):
            cy = edm_func(x, *p)
            ty += cy

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.plot(x, ty)
                fig.tight_layout()
                plt.show()

        if using_hybrid:
            cft1 = cfs.compute_cf(x, ty)
            return np.log(np.sum((ty - y)**2)) + CFSE_WEIGHT*np.log(np.sum((cft1 - cft0)**2))
        elif using_cfs:
            cft1 = cfs.compute_cf(x, ty)
            return np.sum((cft1 - cft0)**2)
        else:
            return np.sum((ty - y)**2)

    init_edms = guess_edm_initial_params(x, y)
    u = 0.5
    edm_params_list = []
    for p in init_edms:
        edm_params_list.append((u, *p))

    params = np.array(edm_params_list).flatten()
    # objective_edm_cfs(params, debug=True)

    ret_edm = minimize(objective_edm_cfs, params)
    # objective_edm_cfs(ret_edm.x, debug=True)

    return ret_egha.x.reshape(shape), ret_emga.x.reshape(shape), ret_edm.x.reshape(edm_shape)
