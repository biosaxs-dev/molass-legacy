"""
    Models/Stochastic/LognormalPore.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Tentative import Model
from molass_legacy.Models.Stochastic.LognormalPoreColumn import LognormalPoreColumn
from molass_legacy.Models.ElutionModelUtils import x_from_height_ratio_impl
from molass_legacy.Models.Stochastic.LognormalPoreColumn import lognormal_pore_func

class LognormalPore(Model):
    def __init__(self, **kwargs):
        self.column = LognormalPoreColumn()
        super(LognormalPore, self).__init__(self.column.compute_curve, **kwargs)

    def get_name(self):
        return "STCln"

    def is_traditional(self):
        return False

    def guess_multiple(self, x, y, num_peaks, debug=False):
        if debug:
            from importlib import reload
            import Models.Stochastic.LognormalPoreImpl as LognormalPoreImpl
            reload(LognormalPoreImpl)
        from molass_legacy.Models.Stochastic.LognormalPoreImpl import guess_multiple_impl
        return guess_multiple_impl(x, y, num_peaks, debug=debug)

    def eval(self, params=None, x=None):
        return self.func(x, *params)

    def x_from_height_ratio(self, ecurve, ratio, params):
        return x_from_height_ratio_impl(lognormal_pore_func, ecurve, ratio, *params, needs_ymax=True, full_params=True)

    def get_params_string(self, params):
        return 'scale=%g, N=%g, T=%g, rg=%g, x0=%g' % tuple(params)

    def adjust_to_xy(self, params_list, x, y, props=None, devel=False):
        if props is None:
            areas = []
            for p in params_list:
                cy = lognormal_pore_func(x, *p)
                areas.append(np.sum(cy))
            props = np.array(areas)/np.sum(areas)

        print("props=", props)

        if devel:
            def plot_with_params_list(title, params_list):
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title(title)
                    ax.plot(x, y)
                    cy_list = []
                    for p in params_list:
                        cy = lognormal_pore_func(x, *p)
                        cy_list.append(cy)
                        ax.plot(x, cy, ":")
                    ty = np.sum(cy_list, axis=0)
                    ax.plot(x, ty, ":", color="red")
                    fig.tight_layout
                    plt.show()
            plot_with_params_list("before conversion", params_list)

        params_array = np.array(params_list)

        def objective(p):
            cy_list = []
            areas = []
            for params in p.reshape(params_array.shape):
                cy = lognormal_pore_func(x, *params)
                cy_list.append(cy)
                areas.append(np.sum(cy))
            ty = np.sum(cy_list, axis=0)
            props_ = np.array(areas)/np.sum(areas)
            fv = np.sum((ty - y)**2) + np.sum((props_ - props)**2)
            return fv

        ret = minimize(objective, params_array.flatten(), method='Nelder-Mead')
        print("ret.success=", ret.success)
        converted_array = ret.x.reshape(params_array.shape)

        if devel:
            plot_with_params_list("after conversion", converted_array)
        return converted_array    

def lognormalpore_test_impl(advanced_frame, **kwargs):
    editor = advanced_frame.editor
    editor_frame = editor.get_current_frame()
    model = editor_frame.model
    print("lognormalpore_test_impl", model.get_name(), model.__class__)
    params_array = editor.get_current_params_array()

    fx = editor_frame.fx
    x = editor_frame.x
    y = editor_frame.y
    uv_y = editor_frame.uv_y

    lnpore = LognormalPore()
    params_array = lnpore.guess_multiple(fx, y, 3, debug=True)
    if params_array is not None:
        pass