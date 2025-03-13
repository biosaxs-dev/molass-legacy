"""
    Batch.BatchMappingImpl.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ModelUtils import compute_cy_list, compute_area_props
from molass_legacy.QuickAnalysis.ModeledPeaks import get_modeled_peaks_impl
from molass_legacy.Peaks.PeakParamsSet import PeakParamsSet

def get_mapped_peaks(model, src_peaks, src_mapping_indeces, tgt_x, tgt_y, tgt_mapping_peaks):
    X = src_peaks[src_mapping_indeces,1]
    Y = tgt_mapping_peaks[:,1]
    slope, intercept = linregress(X, Y)[0:2]
    scale = np.average(tgt_mapping_peaks[:,0]/src_peaks[src_mapping_indeces,1])
    target_params = []
    for h, m, s, t, a in src_peaks:
        h_ = h*scale
        m_ = m*slope + intercept
        s_ = s*slope
        t_ = t*slope
        a_ = a*slope
        target_params.append((h_, m_, s_, t_, a_))
    target_params = np.array(target_params)
    temp_params = target_params.copy()

    def scale_objective(scales):
        cy_list = []
        for params, s in zip(temp_params, scales):
            params[0] = s
            cy = model(tgt_x, params)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        return np.sum((ty - tgt_y)**2)

    res = minimize(scale_objective, temp_params[:,0])
    temp_params[:,0] = res.x

    max_sigma = np.max(temp_params[:,2])*2
    max_tau = np.max(temp_params[:,3])*2

    def full_objective(p):
        params_array = p.reshape(temp_params.shape)
        cy_list = []
        for params in params_array:
            cy = model(tgt_x, params)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)

        max_sigma_ = np.max(params_array[:,2])
        max_tau_ = np.max(params_array[:,3])

        # penalty: to avoid too large sigma and tau
        penalty = max(0, max_sigma_ - max_sigma)**2 + max(0, max_tau_ - max_tau)**2

        return np.log(np.sum((ty - tgt_y)**2)) + np.log(1+penalty)*10

    res = minimize(full_objective, temp_params.flatten())

    return res.x.reshape(target_params.shape)

def show_mapped_result_impl(self, uv_x, uv_y, xr_x, xr_y, debug=False):
    mapping_src = self.mapping_src
    x = mapping_src.xr_x
    model = mapping_src.model
    cy_list = compute_cy_list(model, x, mapping_src.xr_peaks)
    props = compute_area_props(cy_list)
    print("props=", props)
    num_mainpeaks = 2
    indeces = np.argpartition(props, -num_mainpeaks)[-num_mainpeaks:]
    print(indeces)

    a, b = self.get_pre_recog_mapping_params()
    uv_peaks, xr_peaks = get_modeled_peaks_impl(a, b, uv_x, uv_y, xr_x, xr_y, 2, exact_num_peaks=2, affine=True)
    tgt_uv_peaks = get_mapped_peaks(model, mapping_src.uv_peaks, indeces, uv_x, uv_y, uv_peaks)
    tgt_xr_peaks = get_mapped_peaks(model, mapping_src.xr_peaks, indeces, xr_x, xr_y, xr_peaks)

    if debug:
        with plt.Dp():
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
            fig.suptitle("Mapped Decomposition Result", fontsize=20)
            ax1, ax2 = axes[0,:]
            ax1.set_title("20220716/OA_ALD_201 (UV)", fontsize=16)
            ax2.set_title("20220716/OA_ALD_201 (XR)", fontsize=16)
            mapping_src.draw(axes_info=(ax1, ax2))
            ax1, ax2 = axes[1,:]
            ax1.set_title("20220716/OA_ALD_202 (UV)", fontsize=16)
            ax2.set_title("20220716/OA_ALD_202 (XR)", fontsize=16)
            ax1.plot(uv_x, uv_y, color='blue')
            for p in tgt_uv_peaks:
                cy = model(uv_x, p)
                ax1.plot(uv_x, cy, ":")
            ax2.plot(xr_x, xr_y, color='orange')
            for p in tgt_xr_peaks:
                cy = model(xr_x, p)
                ax2.plot(xr_x, cy, ":")
            fig.tight_layout()
            plt.show()

    self.peak_params_set = PeakParamsSet(tgt_uv_peaks, tgt_xr_peaks, a, b) 
    return tgt_uv_peaks, tgt_xr_peaks