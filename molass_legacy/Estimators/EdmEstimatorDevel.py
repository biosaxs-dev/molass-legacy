"""
    Estimators.EdmEstimatorDevel.py
"""
import numpy as np
import matplotlib.pyplot as plt
from .EghEstimator import EghEstimator

class MappingProxy:
    def __init__(self, a, b):
        self.slope = a
        self.intercept = b

class SsdProxy:
    def __init__(self, mapping):
        self.xr = None
        self.uv = None
        self.mapping = mapping

    def get_mapping(self):
        return self.mapping

class EdmEstimatorDevel(EghEstimator):
    def __init__(self, editor, n_components):
        self.n_components = n_components
        EghEstimator.__init__(self, editor)

    def estimate_params(self, debug=False):
        from molass.LowRank.Decomposition import Decomposition
        from molass.LowRank.ComponentCurve import ComponentCurve
        from molass.PlotUtils.DecompositionPlot import plot_elution_curve

        init_xr_params, init_xr_baseparams, temp_rgs, init_mapping, init_uv_heights, init_uv_baseparams, init_mappable_range, seccol_params = self.estimate_egh_params()
        print("Initial XR Params:", init_xr_params)
        uv_curve, xr_curve = self.ecurves
        xr_x = xr_curve.x

        egh_xr_ccurves = []
        for i, param in enumerate(init_xr_params):
            print(f"  Component {i}: {param}")
            egh_xr_ccurves.append(ComponentCurve(xr_x, param))

        uv_x = uv_curve.x
        a, b = init_mapping
        egh_uv_ccurves = []
        for i, (h, params) in enumerate(zip(init_uv_heights, init_xr_params)):
            tR = a * params[1] + b
            sigma = a * params[2]
            tau = a * params[3]
            uv_params = np.array([h, tR, sigma, tau])
            egh_uv_ccurves.append(ComponentCurve(uv_x, uv_params))

        ssd = SsdProxy(MappingProxy(*init_mapping))
        xr_icurve = xr_curve    # compatible?
        uv_icurve = uv_curve    # compatible?

        decomposition = Decomposition(ssd, xr_icurve, egh_xr_ccurves, uv_icurve, egh_uv_ccurves)
        if debug:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
            fig.suptitle("EGH Initial Parameters by EdmEstimatorDevel")
            plot_elution_curve(ax1, uv_curve, egh_uv_ccurves)
            plot_elution_curve(ax2, xr_curve, egh_xr_ccurves)
            plt.show()

        edm_decomposition = decomposition.optimize_with_model('EDM')
        edm_uv_ccurves = edm_decomposition.uv_ccurves
        edm_xr_ccurves = edm_decomposition.xr_ccurves
        if debug:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
            fig.suptitle("EDM Initial Parameters by EdmEstimatorDevel")
            plot_elution_curve(ax1, uv_curve, edm_uv_ccurves)
            plot_elution_curve(ax2, xr_curve, edm_xr_ccurves)
            plt.show()
 
        xr_params_list = []
        for ccurve in edm_xr_ccurves:
            xr_params_list.append(ccurve.params)
        xr_params = np.array(xr_params_list)
        Tz = np.average(xr_params[:,0])
        nc = self.n_components - 1
        uv_w = np.array([uv_curve.max_y/xr_curve.max_y] * nc)
        init_params = np.concatenate([xr_params.flatten(), init_xr_baseparams, temp_rgs, init_mapping, uv_w, init_uv_baseparams, init_mappable_range, [Tz]])
        return init_params