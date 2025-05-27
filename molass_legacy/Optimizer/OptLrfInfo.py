"""
    Optimizer.OptLrfInfo.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

VARY_SMALL_VALUE = 1e-8

def get_ratio_cy_list(y, cy_list):
    basline = cy_list[-1]
    y_ = y - basline
    safe_ty = np.sum(cy_list[:-1], axis=0)   # excluding baseline
    where_small = np.abs(safe_ty) < VARY_SMALL_VALUE
    safe_ty[where_small] = VARY_SMALL_VALUE
    ret_list = []
    for cy in cy_list[:-1]:
        ret_list.append(y_*cy/safe_ty)
    ret_list.append(basline)    # basseline is not affected by ratio
    return ret_list

class OptLrfInfo:
    def __init__(self, Pxr, Cxr, Puv, Cuv, mapped_UvD,
                    qv, xrD, xrE,
                    x, y, xr_ty, scaled_xr_cy_array,
                    uv_x, uv_y, uv_ty, scaled_uv_cy_array,
                    composite):
        self.matrices = (Pxr, Cxr, Puv, Cuv, mapped_UvD)
        self.qv = qv
        self.xrD = xrD
        self.xrE = xrE
        self.x = x
        self.y = y
        self.xr_ty = xr_ty
        self.scaled_xr_cy_array = scaled_xr_cy_array
        self.uv_x = uv_x
        self.uv_y = uv_y
        self.uv_ty = uv_ty
        self.scaled_uv_cy_array = scaled_uv_cy_array
        self.composite = composite

    def get_num_substantial_components(self):
        return self.composite.get_num_substantial_components()

    def get_xr_cy_list(self):
        return list(self.scaled_xr_cy_array)

    def get_xr_proportions(self, exclude_baseline=True, debug=False):
        stop = -1 if exclude_baseline else None
        areas = []
        for cy in self.scaled_xr_cy_array[:stop]:
            areas.append(np.sum(cy))

        if debug:
            x = self.x
            y = self.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_xr_proportions")
                ax.plot(x, y)
                areas = []
                for cy in self.scaled_xr_cy_array:
                    ax.plot(x, cy, ":")
                    areas.append(np.sum(cy))
                areas = np.array(areas)
                props1 = areas/np.sum(areas)
                props2 = areas[:-1]/np.sum(areas[:-1])
                print(props1, props2)
                # with ALD002
                # [0.70246131 0.1551997  0.14233899] [0.81904307 0.18095693]
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                tx = xmin*0.9 + xmax*0.1
                ty = ymin*0.3 + ymax*0.7
                dy = (ymax - ymin)*0.1
                ax.text(tx, ty, "props1=%s" % str(props1))
                ax.text(tx, ty-dy, "props2=%s" % str(props2))
                fig.tight_layout()
                plt.show()

        return np.array(areas)/np.sum(areas)

    def get_uv_cy_list(self):
        return list(self.scaled_uv_cy_array)

    def get_uv_proportions(self, exclude_baseline=True):
        stop = -1 if exclude_baseline else None
        areas = []
        for cy in self.scaled_uv_cy_array[:stop]:
            areas.append(np.sum(cy))
        return np.array(areas)/np.sum(areas)

    def estimate_xr_peak_points(self, elutions=None):
        if elutions is None:
            elutions = self.scaled_xr_cy_array
        return self.composite.estimate_peak_points(self.x, elutions)

    def estimate_uv_peak_points(self, elutions=None):
        if elutions is None:
            elutions = self.scaled_uv_cy_array
        return self.composite.estimate_peak_points(self.uv_x, elutions)

    def get_valid_rgs(self, rg_params):
        return self.composite.get_valid_rgs(rg_params)

    def get_scaled_xr_params(self, xr_params, debug=True):
        x = self.x
        ret_params = xr_params.copy()
        for k, m in enumerate(xr_params[:,1]):
            cy = self.scaled_xr_cy_array[k]
            ret_params[k,0] = cy[int(m - x[0])]

        if debug:
            from molass_legacy.Models.ElutionCurveModels import egh
            y = self.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_scaled_xr_params")
                ax.plot(x, y)
                cy_list = []
                for h, m, s, t in ret_params:
                    cy = egh(x, h, m, s, t)
                    cy_list.append(cy)
                    ax.plot(x, cy, ":")
                ty = np.sum(cy_list, axis=0)
                ax.plot(x, ty, ":", color="red")
                fig.tight_layout()
                plt.show()
        return ret_params

    def compute_rg_params(self, devel=True):
        if devel:
            from importlib import reload
            import Optimizer.GuinierRg
            reload(Optimizer.GuinierRg)
        from molass_legacy.Optimizer.GuinierRg import compute_rg_params_impl

        return compute_rg_params_impl(self)

    def update_optimizer(self, optimizer, params, devel=True):
        if devel:
            from importlib import reload
            import Optimizer.RgSecUpdater
            reload(Optimizer.RgSecUpdater)
        from molass_legacy.Optimizer.RgSecUpdater import update_optimizer_impl

        rg_params, rg_qualities = self.compute_rg_params()
        self.rg_params = rg_params
        self.rg_qualities = rg_qualities

        return update_optimizer_impl(optimizer, params, rg_params, rg_qualities)
